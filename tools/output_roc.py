"""Output-referenced evaluation: what gets muted, and what does it cost?

The output is delayed by L, so the mute decision for output frame k may use audio from the
past AND up to L after frame k. Each frame is classified on its own (no blind hold), and the
cost of an error depends on what it hits:

  gulp suppressed      >= SUPPRESSED_FRAC of the gulp's ENERGY muted (what is left is at
                       least 10 dB down); its near-floor tails are inaudible either way
  onset leak           time from gulp onset to its first muted frame
  speech cut           a run of muted ACTIVE H0 frames (gaps < MERGE_MS merged): one per
                       clipped syllable or breath; reported per hour of lecture audio
  muted active audio   seconds per hour of active H0 audio muted
  muting quiet frames  free: nobody hears silence being turned down

'Active' = at least ACTIVE_DB above the lecture's room floor (10th percentile level). Gulp
start and end are measured from the audio inside each label, since review-tool labels inherit
the candidate window and are wider than the sound.

Known bias: unlabelled gulps are in H0, so muting them counts as a speech cut.

Usage: .venv\\Scripts\\python tools\\output_roc.py
"""
import json
import os

import numpy as np

import features
import gulpdata
from find_candidates import fit_logreg, logit
from latency_roc import mean_n

LOOKAHEADS_MS = [25, 50, 100, 150, 200]
PAST = 30               # frames of past context (300 ms)
NEAR = 5                # first frames of the future window, the 'now' part
FLOOR_N = 100           # running-minimum floor over the second before the frame
DIGITAL_SILENCE_DB = -100.0
ACTIVE_DB = 10.0
EDGE_RISE = 0.1         # gulp start/end: where the level crosses 10% of background -> peak
EDGE_MARGIN = 5         # frames around measured gulp edges excluded from H0 (edge uncertainty)
# Training positives are only the gulp's core (level at least half-way from background to peak,
# in dB). Its near-floor tails look like room noise; training on them teaches 'quiet = gulp'.
CORE_RISE = 0.5
SUPPRESSED_FRAC = 0.9
MERGE_MS = 50
N_FOLDS = 6
NEG_PER_LECTURE = 4000
HARD_NEG_WEIGHT = 5.0
Z_CLIP = 4.0
CUT_TARGETS = [1, 5, 20]
DELTAS = [0.0, 4.0, 8.0]   # hysteresis widths tried (log-odds); 0 = plain per-frame threshold
HOP_S = features.HOP / features.FS
FRAME_S = features.WIN / features.FS
OUT = os.path.join(gulpdata.ROOT, 'results')
PADF = 400              # edge padding for window statistics, in frames


def n_future(L_ms):
    """Frames that start at k and have fully ended by (start of frame k) + L."""
    return max(1, int(np.floor((L_ms / 1000 - FRAME_S) / HOP_S + 1e-9)) + 1)


class Win:
    """Window statistics over frames k+off .. k+off+n-1 for every k (edge-padded)."""

    def __init__(self, a):
        self.T = len(a)
        self.p = np.pad(a, [(PADF, PADF)] + [(0, 0)] * (a.ndim - 1), mode='edge').astype(np.float64)
        self.c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(self.p, axis=0)])

    def mean(self, off, n):
        st = np.arange(self.T) + off + PADF
        return (self.c[st + n] - self.c[st]) / n

    def reduce(self, off, n, fn):
        v = np.lib.stride_tricks.sliding_window_view(self.p, n, axis=0)
        return fn(v[np.arange(self.T) + off + PADF], axis=-1)


def frame_features(f, L_ms):
    """Features for the mute decision of every output frame, plus an 'enabled' mask."""
    nf = n_future(L_ms)
    nn = min(NEAR, nf)
    bands, total, harm, flat = f['bands'], f['total'], f['harm'], f['flat']
    flux = np.concatenate([[0], np.maximum(np.diff(bands, axis=0), 0).sum(axis=1)])
    B, Tt, H, Fl, Fx = Win(bands), Win(total), Win(harm), Win(flat), Win(flux)
    floor = Win(mean_n(total, 5)).reduce(-FLOOR_N, FLOOR_N, np.min)
    p95 = np.percentile(total, 95)
    dig = Win((total <= DIGITAL_SILENCE_DB).astype(np.float64)).mean(-2 * FLOOR_N, 2 * FLOOR_N + nf) > 0

    mb = B.mean(0, nf)
    cols = [mb - mb.mean(axis=1, keepdims=True)]                    # spectral shape of the future
    for off, n in [(-PAST, PAST), (0, nn), (0, nf)]:                # past, now, future
        cols += [Tt.mean(off, n) - p95, H.mean(off, n), Fl.mean(off, n)]
    cols += [Tt.reduce(0, nf, np.max) - floor, H.reduce(0, nf, np.max), Fx.reduce(0, nf, np.max),
             np.sqrt(np.maximum(Win(total.astype(np.float64) ** 2).mean(0, nf) - Tt.mean(0, nf) ** 2, 0)),
             Tt.mean(0, nn) - floor, Tt.mean(-PAST, PAST) - floor, Tt.reduce(-PAST, PAST, np.max) - floor,
             floor - p95]
    X = np.column_stack(cols).astype(np.float32)
    return X, ~dig


def gulp_edges(total, s, e):
    """First and last frame inside [s, e] at least EDGE_RISE of the way from background to peak,
    and the frames of its core (at least CORE_RISE of the way)."""
    a, b = int(round(s / HOP_S)), max(int(round(e / HOP_S)), int(round(s / HOP_S)) + 1)
    bg = np.median(total[max(0, a - 50):a]) if a > 0 else total[a]
    seg = total[a:b]
    above = np.flatnonzero(seg >= bg + EDGE_RISE * (seg.max() - bg))
    core = a + np.flatnonzero(seg >= bg + CORE_RISE * (seg.max() - bg))
    return a + above[0], a + above[-1], core


def frame_sets(total, labels):
    T = len(total)
    t = np.arange(T) * HOP_S
    gulp = np.zeros(T, bool)
    core = np.zeros(T, bool)
    ignore = np.zeros(T, bool)
    hard = np.zeros(T, bool)
    events = []
    for s, e, lab in labels:
        span = (t >= s) & (t <= e)
        if lab in gulpdata.GULP_LABELS:
            k0, k1, kc = gulp_edges(total, s, e)
            gulp[k0:k1 + 1] = True
            core[kc] = True
            ignore |= span
            ignore[max(0, k0 - EDGE_MARGIN):k1 + EDGE_MARGIN + 1] = True
            events.append({'label': lab, 'k0': k0, 'k1': k1})
        elif lab in gulpdata.IGNORE_LABELS:
            ignore |= span
        elif lab in gulpdata.KEEP_LABELS:
            hard |= span
    active = total >= np.percentile(total, 10) + ACTIVE_DB
    ignore &= ~gulp
    h0 = active & ~gulp & ~ignore
    return events, core, h0, hard & h0


def cut_counts(scores, h0, thresholds, merge):
    """Speech cuts at each threshold: rising edges of the merge-dilated muted-active indicator."""
    x = np.where(h0, scores, -np.inf)
    pad = np.concatenate([np.full(merge - 1, -np.inf), x])
    d = np.lib.stride_tricks.sliding_window_view(pad, merge).max(axis=1)
    prev = np.concatenate([[-np.inf], d[:-1]])
    up = d > prev
    lo, hi = np.sort(prev[up]), np.sort(d[up])
    return np.searchsorted(lo, thresholds, 'right') - np.searchsorted(hi, thresholds, 'right')


def run(L_ms, lectures, feats, labels, baseline=False):
    data = {}
    for lec in lectures:
        X, enabled = frame_features(feats[lec], L_ms)
        ev, gulp, h0, hard = frame_sets(feats[lec]['total'], labels[lec])
        data[lec] = (X, ev, gulp & enabled, h0 & enabled, hard & enabled, enabled)
    lvl_col = 24 + 9   # 'future max level above floor'
    folds = [lectures[i::N_FOLDS] for i in range(N_FOLDS)]
    rng = np.random.default_rng(0)
    scored = {}
    for fold in folds:
        if baseline:
            for lec in fold:
                scored[lec] = data[lec][0][:, lvl_col].astype(np.float64)
        else:
            Xs, ys, ws = [], [], []
            for lec in lectures:
                if lec in fold:
                    continue
                X, ev, gulp, h0, hard, _ = data[lec]
                pos = np.flatnonzero(gulp)
                hn = np.flatnonzero(hard)
                pool = np.flatnonzero(h0 & ~hard)
                neg = rng.choice(pool, min(NEG_PER_LECTURE, len(pool)), replace=False)
                Xs += [X[pos], X[hn], X[neg]]
                ys += [np.ones(len(pos)), np.zeros(len(hn)), np.zeros(len(neg))]
                ws += [np.ones(len(pos)), np.full(len(hn), HARD_NEG_WEIGHT), np.ones(len(neg))]
            X, y, w = np.vstack(Xs).astype(np.float64), np.concatenate(ys), np.concatenate(ws)
            w[y == 1] *= w[y == 0].sum() / w[y == 1].sum()
            mu, sd = X.mean(0), X.std(0) + 1e-6
            beta = fit_logreg(np.clip((X - mu) / sd, -Z_CLIP, Z_CLIP), y, w)
            for lec in fold:
                scored[lec] = logit(beta, np.clip((data[lec][0] - mu) / sd, -Z_CLIP, Z_CLIP))
        for lec in fold:
            scored[lec] = np.where(data[lec][5], scored[lec], -np.inf)

    return scored, data


def hysteresis(s, hi, lo):
    """Mute runs where s > lo that contain at least one frame with s > hi (Schmitt-style)."""
    above = s > lo
    if not above.any():
        return above
    start = above & ~np.concatenate([[False], above[:-1]])
    run_id = np.cumsum(start) - 1
    idx = np.flatnonzero(above)
    rid = run_id[idx]
    has_hi = np.zeros(rid.max() + 1, bool)
    np.logical_or.at(has_hi, rid, s[idx] > hi)
    out = np.zeros_like(above)
    out[idx] = has_hi[rid]
    return out


def evaluate(scored, data, feats, lectures, delta, n_th=120):
    """Sweep the high threshold; the low threshold sits delta (log-odds) below it."""
    hours = sum(len(d[0]) for d in data.values()) * HOP_S / 3600
    act_hours = sum(d[3].sum() for d in data.values()) * HOP_S / 3600
    h0_scores = np.sort(np.concatenate([scored[l][data[l][3]] for l in lectures]))
    pos = np.concatenate([scored[l][data[l][2]] for l in lectures])
    pos = pos[np.isfinite(pos)]
    auc = np.searchsorted(h0_scores, pos, 'left').mean() / len(h0_scores)   # P(gulp frame > H0 frame)
    fin = h0_scores[np.isfinite(h0_scores)]
    th = np.unique(np.quantile(fin, 1 - np.geomspace(1e-2, 1e-7, n_th)))
    merge = int(MERGE_MS / 1000 / HOP_S)
    energy = {}
    for l in lectures:
        tot = feats[l]['total']
        fl = np.percentile(tot, 10)
        # Linear energy above the room floor (only the excess is audible).
        energy[l] = [np.maximum(10 ** ((tot[e['k0']:e['k1'] + 1] - fl) / 10) - 1, 0) + 1e-9 for e in data[l][1]]
    supp, cuts, muted, leak, frac_all = [], [], [], [], []
    for t in th:
        n_cut = n_mut = 0
        fr, lk = [], []
        for l in lectures:
            X, ev, gulp, h0, hard, enabled = data[l]
            m = hysteresis(scored[l], t, t - delta)
            x = (m & h0).astype(np.int64)
            c = np.concatenate([[0], np.cumsum(x)])
            i = np.arange(len(x))
            d = (c[i + 1] - c[np.maximum(0, i - merge + 1)]) > 0     # merge cuts closer than MERGE_MS
            n_cut += int(d[0]) + int((d[1:] & ~d[:-1]).sum())
            n_mut += int(x.sum())
            for e, en in zip(ev, energy[l]):
                seg = m[e['k0']:e['k1'] + 1]
                fr.append((seg * en).sum() / en.sum())
                lk.append((np.argmax(seg) if seg.any() else len(seg)) * HOP_S * 1000)
        frac_all.append(fr)
        supp.append(np.mean(np.array(fr) >= SUPPRESSED_FRAC))
        cuts.append(n_cut / hours)
        muted.append(n_mut * HOP_S / hours)
        leak.append(np.median(lk))
    labels = [e['label'] for l in lectures for e in data[l][1]]
    return {'th': th, 'supp': np.array(supp), 'cuts_per_h': np.array(cuts), 'muted_s_per_h': np.array(muted),
            'leak_ms': np.array(leak), 'frac': np.array(frac_all).T, 'labels': labels,
            'hours': hours, 'act_hours': act_hours, 'n': len(labels), 'auc': auc, 'delta': delta}


def at_cuts(r, target):
    ok = r['cuts_per_h'] <= target
    if not ok.any():
        return None
    return np.flatnonzero(ok)[np.argmax(r['supp'][ok])]


def main():
    lectures = [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]
    feats = {l: features.lecture_features(l) for l in lectures}
    labels = {l: gulpdata.load_labels(l) for l in lectures}
    results = []
    for L in LOOKAHEADS_MS:
        scored, data = run(L, lectures, feats, labels)
        for delta in DELTAS:
            results.append({**evaluate(scored, data, feats, lectures, delta), 'L_ms': L})
        del scored, data
    scored, data = run(100, lectures, feats, labels, baseline=True)
    base = {**evaluate(scored, data, feats, lectures, 0.0), 'L_ms': 100}
    del scored, data
    r0 = results[0]
    print(f'{r0["n"]} gulps; {r0["hours"]:.1f} h of audio, {r0["act_hours"]:.1f} h of it active H0; '
          f'{N_FOLDS}-fold CV by lecture\n')
    print(f'{"detector":<34}{"AUC":>6}' + ''.join(f'{"@ " + str(t) + " cuts/h":>25}' for t in CUT_TARGETS))
    print(f'{"":<40}' + '     suppressed  leak  muted' * len(CUT_TARGETS))
    rows = []
    for r in results + [base]:
        name = (f'logistic, L = {r["L_ms"]} ms, hyst {r["delta"]:g}' if r is not base
                else 'level only, L = 100 ms')
        line = f'{name:<34}{r["auc"]:>6.3f}'
        for t in CUT_TARGETS:
            i = at_cuts(r, t)
            line += ('         -     -      -' if i is None else
                     f'{r["supp"][i]:>14.0%}{r["leak_ms"][i]:>5.0f}ms{r["muted_s_per_h"][i]:>5.1f}s')
        print(line)
        rows.append({'name': name, 'L_ms': r['L_ms'], 'delta': r['delta'], 'supp': r['supp'].tolist(),
                     'cuts_per_h': r['cuts_per_h'].tolist(), 'baseline': r is base})
    print('  (AUC = frame level; hyst = low threshold this many log-odds below the high one; '
          'suppressed = gulps with >= 90% of their energy muted; leak = median ms of each gulp heard before '
          'the first mute; muted = seconds of active H0 audio muted per hour)')

    print('\nsuppressed by severity at 5 cuts/h (best hysteresis per L):')
    best = {}
    for r in results:
        i = at_cuts(r, 5)
        v = r['supp'][i] if i is not None else -1
        if v > best.get(r['L_ms'], (None, -2))[1]:
            best[r['L_ms']] = (r, v)
    for r, _ in best.values():
        i = at_cuts(r, 5)
        if i is None:
            continue
        parts = []
        for lab in ['gulp_low', 'gulp', 'gulp_high', 'gulp_overlap']:
            m = np.array([l == lab for l in r['labels']])
            if m.any():
                parts.append(f'{lab} {np.mean(r["frac"][m, i] >= SUPPRESSED_FRAC):.0%} of {m.sum()}')
        print(f'  L = {r["L_ms"]:>3} ms, hyst {r["delta"]:g}: ' + ', '.join(parts))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'output_roc.json'), 'w', encoding='utf-8') as f:
        json.dump({'n_gulps': r0['n'], 'hours': r0['hours'], 'curves': rows}, f)
    print(f'\ncurves saved to {os.path.relpath(os.path.join(OUT, "output_roc.json"), gulpdata.ROOT)}')


if __name__ == '__main__':
    main()
