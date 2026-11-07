"""Rank 300 ms windows by how much they look like the labelled gulps; write candidates.json.

This is a labelling assistant, not the detector. It sees the whole 300 ms window, while the
real-time detector will only see what fits in its lookahead, so its scores say nothing
about final performance. Test lectures (037-040) are never trained on or searched.

Model: L2-regularised logistic regression on per-window summary features, trained with
labelled gulps as H1 and random unlabelled windows plus reviewed H0 sounds as H0. Random
windows may contain unlabelled gulps (label noise), which is fine for ranking.

Usage: .venv\\Scripts\\python tools\\find_candidates.py [--max 300]
"""
import argparse
import json
import os
import sys

import numpy as np

import features
import gulpdata

W = 30          # window length in frames (300 ms, about one gulp)
STEP = 5        # window hop in frames (50 ms)
FPS = features.FS / features.HOP   # frames per second (100)
NMS_S = 1.0     # suppress peaks closer than this to a stronger one
HARD_NEG_WEIGHT = 5.0   # reviewed look-alikes matter more than random windows
MIN_ABOVE_FLOOR_DB = 3.0  # a gulp is an event above the background; labelled ones sit 7.5+ dB above
Z_CLIP = 4.0            # clip standardised features so the linear model cannot extrapolate,
                        # e.g. score digital silence as 'extremely gulp-like' for being quiet
RNG = np.random.default_rng(0)     # fixed seed: same labels -> same candidate list


def _win_mean(a, starts):
    c = np.concatenate([np.zeros((1,) + a.shape[1:], a.dtype), np.cumsum(a, axis=0, dtype=np.float64)])
    return (c[starts + W] - c[starts]) / W


def _win_max(a, starts):
    return np.lib.stride_tricks.sliding_window_view(a, W, axis=0)[starts].max(axis=-1)


FEATURE_NAMES = ([f'shape_{i}' for i in range(features.N_BANDS)] +
                 ['level_vs_speech', 'level_vs_floor', 'level_std',
                  'harm_mean', 'harm_max', 'flat_mean', 'flux_mean', 'flux_max'])


def window_features(f):
    """One row per window start (every STEP frames)."""
    bands, total = f['bands'], f['total']
    starts = np.arange(0, len(total) - W + 1, STEP)
    mb = _win_mean(bands, starts)
    shape = mb - mb.mean(axis=1, keepdims=True)    # spectral shape, independent of loudness
    mt = _win_mean(total, starts)
    p10, p95 = np.percentile(total, [10, 95])      # background floor and loud-speech level
    std = np.sqrt(np.maximum(_win_mean(total.astype(np.float64) ** 2, starts) - mt ** 2, 0))
    flux = np.concatenate([[0], np.maximum(np.diff(bands, axis=0), 0).sum(axis=1)]).astype(np.float32)
    X = np.column_stack([shape, mt - p95, mt - p10, std,
                         _win_mean(f['harm'], starts), _win_max(f['harm'], starts),
                         _win_mean(f['flat'], starts),
                         _win_mean(flux, starts), _win_max(flux, starts)])
    return X.astype(np.float32), starts


def fit_logreg(X, y, w, lam=3.0, iters=30):
    """Weighted L2 logistic regression by Newton's method (IRLS); bias is not penalised."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    beta = np.zeros(Xb.shape[1])
    R = lam * np.eye(Xb.shape[1])
    R[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        g = Xb.T @ (w * (p - y)) + R @ beta
        H = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-6:
            break
    return beta


def logit(beta, X):
    """Decision value z = log-odds. Used for ranking because p = sigmoid(z) saturates at 1
    for the strongest windows, which would make their order arbitrary."""
    return X @ beta[:-1] + beta[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=400, help='number of candidates to keep')
    ap.add_argument('--per-lecture', type=int, default=20,
                    help='cap per lecture; held-out tests put most known gulps in the top 20')
    args = ap.parse_args()

    train = [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]
    data = {}
    for i, lec in enumerate(train):
        print(f'\rfeatures {i + 1}/{len(train)} ({lec})', end='', flush=True)
        f = features.lecture_features(lec)
        X, starts = window_features(f)
        centre = (starts + W / 2) / FPS
        data[lec] = (X, starts, centre, gulpdata.load_labels(lec), float(f['hf_fraction']))
    print()

    hf = np.array([d[4] for d in data.values()])
    print(f'energy above 8 kHz: worst lecture {10 * np.log10(hf.max()):.1f} dB, '
          f'median {10 * np.log10(np.median(hf)):.1f} dB of total')

    # Training set. A window is H1 when its centre lies inside a gulp label.
    Xs, ys, ws = [], [], []
    n_pos = n_hard = 0
    neg_per_lecture = 1000
    for lec, (X, starts, centre, labels, _) in data.items():
        near_any = np.zeros(len(centre), bool)
        for s, e, lab in labels:
            inside = (centre >= s) & (centre <= e)
            near_any |= (centre >= s - 1.0) & (centre <= e + 1.0)
            if lab in gulpdata.GULP_LABELS:
                Xs.append(X[inside]); ys.append(np.ones(inside.sum())); ws.append(np.ones(inside.sum()))
                n_pos += inside.sum()
            elif lab in gulpdata.KEEP_LABELS:
                Xs.append(X[inside]); ys.append(np.zeros(inside.sum()))
                ws.append(np.full(inside.sum(), HARD_NEG_WEIGHT))
                n_hard += inside.sum()
        pool = np.flatnonzero(~near_any)
        pick = RNG.choice(pool, min(neg_per_lecture, len(pool)), replace=False)
        Xs.append(X[pick]); ys.append(np.zeros(len(pick))); ws.append(np.ones(len(pick)))
    X, y, w = np.vstack(Xs).astype(np.float64), np.concatenate(ys), np.concatenate(ws)
    if n_pos == 0:
        sys.exit('no gulp labels found')
    w[y == 1] *= w[y == 0].sum() / w[y == 1].sum()   # balance the classes
    mu, sd = X.mean(0), X.std(0) + 1e-6
    norm = lambda A: np.clip((A - mu) / sd, -Z_CLIP, Z_CLIP)
    beta = fit_logreg(norm(X), y, w)
    print(f'trained on {n_pos} gulp windows, {n_hard} reviewed H0 windows, {int((y == 0).sum()) - n_hard} random windows')
    top = np.argsort(-np.abs(beta[:-1]))[:6]
    print('strongest features:', ', '.join(f'{FEATURE_NAMES[i]} ({beta[i]:+.2f})' for i in top))

    # Score every window, then keep local peaks (non-maximum suppression).
    peaks, known = [], []
    for lec, (Xl, starts, centre, labels, _) in data.items():
        z = logit(beta, norm(Xl))
        z[Xl[:, FEATURE_NAMES.index('level_vs_floor')] < MIN_ABOVE_FLOOR_DB] = -np.inf
        for s, e, lab in labels:
            if lab in gulpdata.GULP_LABELS:
                inside = (centre >= s) & (centre <= e)
                known.append(z[inside].max() if inside.any() else -np.inf)
        order = np.argsort(-z)
        taken = np.zeros(len(z), bool)
        n_step = int(NMS_S * FPS / STEP)
        kept = 0
        for i in order:
            if z[i] < 0 or kept >= args.per_lecture:     # z < 0 means p < 0.5
                break
            if taken[i]:
                continue
            taken[max(0, i - n_step):i + n_step + 1] = True
            # Region: neighbouring windows whose odds are within e^3 (~20x) of the peak.
            floor = max(0.0, z[i] - 3.0)
            a = i
            while a > 0 and i - a < n_step and z[a - 1] >= floor:
                a -= 1
            b = i
            while b < len(z) - 1 and b - i < n_step and z[b + 1] >= floor:
                b += 1
            t0, t1 = starts[a] / FPS, (starts[b] + W) / FPS
            # Skip anything already labelled by you or decided in the review tool.
            if any(t0 < e and s < t1 for s, e, _ in labels):
                continue
            peaks.append({'lec': lec, 'start': round(t0, 3), 'end': round(t1, 3), 'score': round(float(z[i]), 2)})
            kept += 1

    peaks.sort(key=lambda c: -c['score'])
    peaks = peaks[:args.max]
    for r, c in enumerate(peaks, 1):
        c['id'] = f"{c['lec']}_{c['start']:.3f}"
        c['rank'] = r
    known = np.array(known)
    cut = peaks[-1]['score'] if peaks else np.inf
    print(f'known gulps scoring above the last candidate (z = {cut:.2f}): {(known >= cut).sum()}/{len(known)} '
          '(training data, so this is optimistic)')

    os.makedirs(gulpdata.REVIEW, exist_ok=True)
    out = os.path.join(gulpdata.REVIEW, 'candidates.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'model': {'features': FEATURE_NAMES, 'n_pos_windows': int(n_pos)}, 'candidates': peaks}, fh, indent=1)
    print(f'{len(peaks)} candidates written to {os.path.relpath(out, gulpdata.ROOT)}')


if __name__ == '__main__':
    main()
