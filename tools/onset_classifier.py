"""Two-stage detector: find sound onsets, then classify only the first N ms after each one.

Stage 1 (onset detector, causal): an onset is a frame whose level is at least ONSET_DB above
the tracked room floor after QUIET_MS of audio that stayed below that line. In hardware this
is a comparator and a counter.

Stage 2 (onset classifier): the spectro-temporal patch from PRE frames before the onset to N ms
after it (24 bands per 10 ms frame, plus harmonicity and spectral flux), not averaged over
time, so short 'click' structure survives. Logistic regression on the flattened patch.

Why: frame-by-frame classification compared every 10 ms of audio with gulps, about one in
1700 frames. Here a gulp onset competes only with other onsets, a far better base rate.
The decision is ready N ms after the onset, so N is the lookahead L.

Metrics (event level, lecture-grouped cross-validation):
  P_D  = labelled gulps whose onset was found AND classified as gulp (stage-1 misses count)
  FA/h = non-gulp onsets classified as gulp, per hour of audio

Known bias: unlabelled gulps' onsets are negatives, so FA/h is overestimated.

Usage: .venv\\Scripts\\python tools\\onset_classifier.py
"""
import numpy as np

import features
import gulpdata
from find_candidates import fit_logreg, logit
from latency_roc import mean_n
from output_roc import gulp_edges, n_future

ONSET_DB = 6.0
QUIET_MS = 100
FLOOR_N = 100
DIGITAL_SILENCE_DB = -100.0
PRE = 3                 # frames of context before the onset (30 ms)
LOOKAHEADS_MS = [25, 50, 75, 100]
LAMBDAS = [10.0, 100.0, 1000.0]
HARD_NEG_WEIGHT = 3.0
N_FOLDS = 6
FA_TARGETS = [1, 5, 20]
HOP_S = features.HOP / features.FS


def detect_onsets(total):
    """Onset frame indices and the floor at each frame."""
    T = len(total)
    sm = mean_n(total, 5)
    run_min = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.full(FLOOR_N, np.inf), sm]), FLOOR_N).min(axis=1)[:T]   # frames k-100..k-1
    floor = np.where(np.isfinite(run_min), run_min, sm)
    above = total >= floor + ONSET_DB
    q = int(QUIET_MS / 1000 / HOP_S)
    c = np.concatenate([[0], np.cumsum(above)])
    k = np.arange(T)
    quiet_before = (c[k] - c[np.maximum(0, k - q)]) == 0
    dig = np.concatenate([[0], np.cumsum(total <= DIGITAL_SILENCE_DB)])
    enabled = (k >= 2 * FLOOR_N) & ((dig[k + 1] - dig[np.maximum(0, k - 2 * FLOOR_N)]) == 0)
    return np.flatnonzero(above & quiet_before & enabled), floor


def patches(f, onsets, floor, n_after):
    """Flattened patch per onset: frames k-PRE .. k+n_after-1."""
    T = len(f['total'])
    flux = np.concatenate([[0], np.maximum(np.diff(f['bands'], axis=0), 0).sum(axis=1)])
    per_frame = np.column_stack([f['bands'], f['harm'], flux])          # (T, 26)
    p95 = np.percentile(f['total'], 95)
    idx = onsets[:, None] + np.arange(-PRE, n_after)[None, :]
    idx = np.clip(idx, 0, T - 1)
    P = per_frame[idx].copy()                                            # (n, frames, 26)
    P[:, :, :features.N_BANDS] -= floor[onsets][:, None, None]          # bands relative to the floor
    extra = np.column_stack([floor[onsets] - p95])                      # room level vs loud speech
    return np.hstack([P.reshape(len(onsets), -1), extra]).astype(np.float64)


def lecture_onsets(f, labels):
    """Onsets with labels: 1 = gulp onset, 0 = other onset, -1 = ignore. Plus gulp list."""
    total = f['total']
    on, floor = detect_onsets(total)
    t = on * HOP_S
    y = np.zeros(len(on), int)
    w = np.ones(len(on))
    gulps = []
    for s, e, lab in labels:
        inside = (t >= s - 0.05) & (t <= e)
        if lab in gulpdata.GULP_LABELS:
            k0, k1, _ = gulp_edges(total, s, e)
            near = np.flatnonzero((on >= k0 - 5) & (on <= k1))
            y[inside] = -1                               # later onsets inside the gulp: ignore
            if len(near):
                y[near[0]] = 1
            gulps.append({'label': lab, 'onset_index': int(near[0]) if len(near) else None})
        elif lab in gulpdata.IGNORE_LABELS:
            y[inside] = -1
        elif lab in gulpdata.KEEP_LABELS:
            w[inside & (y == 0)] = HARD_NEG_WEIGHT
    return on, floor, y, w, gulps


def cv_scores(N, lam, lectures, feats, base):
    """Cross-validated classifier score of every onset, per lecture (no lecture scores itself)."""
    n_after = n_future(N)
    X = {l: patches(feats[l], base[l][0], base[l][1], n_after) for l in lectures}
    out = {}
    for fold in [lectures[i::N_FOLDS] for i in range(N_FOLDS)]:
        tr = [l for l in lectures if l not in fold]
        Xt = np.vstack([X[l][base[l][2] >= 0] for l in tr])
        yt = np.concatenate([base[l][2][base[l][2] >= 0] for l in tr]).astype(float)
        wt = np.concatenate([base[l][3][base[l][2] >= 0] for l in tr])
        wt[yt == 1] *= wt[yt == 0].sum() / wt[yt == 1].sum()
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        beta = fit_logreg(np.clip((Xt - mu) / sd, -4, 4), yt, wt, lam=lam)
        for l in fold:
            out[l] = logit(beta, np.clip((X[l] - mu) / sd, -4, 4))
    return out


def load_all(exclude=None):
    lectures = [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]
    feats = {l: features.lecture_features(l) for l in lectures}
    base = {l: lecture_onsets(feats[l], gulpdata.load_labels(l, exclude)) for l in lectures}
    return lectures, feats, base


def main():
    lectures, feats, base = load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * HOP_S / 3600

    n_gulps = sum(len(b[4]) for b in base.values())
    found = sum(g['onset_index'] is not None for b in base.values() for g in b[4])
    n_neg = sum((b[2] == 0).sum() for b in base.values())
    print(f'{hours:.1f} h, {n_gulps} gulps; stage 1 finds {sum(len(b[0]) for b in base.values()) / hours:.0f} onsets/h')
    print(f'stage 1 recall: {found}/{n_gulps} gulps have a detected onset ({found / n_gulps:.0%}); '
          f'{n_neg} other onsets ({n_neg / hours:.0f}/h) are the competition\n')

    print(f'{"N (= lookahead)":<17}{"lambda":>8}' + ''.join(f'{"P_D @ " + str(t) + "/h":>13}' for t in FA_TARGETS))
    best = {}
    for N in LOOKAHEADS_MS:
        for lam in LAMBDAS:
            scores = cv_scores(N, lam, lectures, feats, base)
            pos_scores, neg_scores, pos_labels = [], [], []
            for l in lectures:
                on, floor, y, w, gulps = base[l]
                z = scores[l]
                neg_scores.append(z[y == 0])
                for g in gulps:
                    pos_scores.append(z[g['onset_index']] if g['onset_index'] is not None else -np.inf)
                    pos_labels.append(g['label'])
            pos, neg = np.array(pos_scores), np.sort(np.concatenate(neg_scores))
            # Threshold for each FA target: allow at most target * hours false alarms.
            row = []
            for t in FA_TARGETS:
                k = int(t * hours)                      # number of negatives allowed above eta
                eta = neg[-k - 1] if k < len(neg) else -np.inf
                row.append((pos > eta).mean())
            print(f'{N:>4} ms{"":<10}{lam:>8g}' + ''.join(f'{v:>13.0%}' for v in row))
            if row[1] > best.get(N, (None, -1, None))[1]:
                eta5 = neg[-int(5 * hours) - 1]
                best[N] = (lam, row[1], {lab: np.mean([p > eta5 for p, q in zip(pos, pos_labels) if q == lab])
                                          for lab in sorted(set(pos_labels))})
    print('\nP_D by severity at 5 FA/h (best lambda per N; picked on the same data, slightly optimistic):')
    for N, (lam, v, sev) in best.items():
        print(f'  N = {N:>3} ms, lambda {lam:g}: ' + ', '.join(f'{k} {v2:.0%}' for k, v2 in sev.items()))


if __name__ == '__main__':
    main()
