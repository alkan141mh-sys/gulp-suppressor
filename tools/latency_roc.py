"""How much of a gulp does a causal detector need to hear? Event-level ROC versus lookahead L.

Real-time constraint. The output is delayed by L. To attenuate a gulp from its first sample
(onset t0), the detector must fire by t0 + L, so at decision time it has heard at most the
first L of the gulp. We model this as a detector whose features come only from a trailing
window that ends at the decision time and spans about L, plus causal context before it.

Metrics (event level, because frame-level rates are misleading at 100 frames/s):
  P_D(eta)  = fraction of gulps whose score exceeds eta at some decision time in [t0, t0 + L]
  FA/h(eta) = threshold crossings per hour in H0 audio, where crossings closer than HOLD_MS
              merge into one (the gain would stay down between them anyway)

Validation: lectures are split into folds; each fold is scored by a model trained on the
other folds only, so no lecture is ever scored by a model that saw it.

Known biases of this development-set estimate (the held-out test lectures fix both later):
  - H0 audio is 'everything not near a label', so unlabelled gulps count as false alarms:
    FA/h is overestimated.
  - Most gulps were found by the candidate ranker, so gulps it cannot find are missing:
    P_D is optimistic for unusual gulps.
  - The speech-level reference is a whole-lecture percentile (non-causal). In hardware a slow
    level tracker would estimate it.

Usage: .venv\\Scripts\\python tools\\latency_roc.py
"""
import json
import os

import numpy as np

import features
import gulpdata
from find_candidates import fit_logreg, logit

LOOKAHEADS_MS = [25, 50, 100, 200]
FLOOR_MS = 1000         # background = running minimum over this long before the window
FLOOR_SMOOTH = 5        # frames averaged before taking the minimum (single frames dip too low)
# Digital silence (file starts, paused playback) sits ~150 dB below the room noise, so any sound
# after it looks like an enormous rise. The detector is disabled while the floor window holds it.
DIGITAL_SILENCE_DB = -100.0
HOLD_MS = 300           # firings closer than this count as one false alarm
GUARD_S = 1.0           # H0 audio excludes this much around every gulp or 'unsure' label
N_FOLDS = 6
NEG_PER_LECTURE = 3000
HARD_NEG_WEIGHT = 5.0
Z_CLIP = 4.0
FA_TARGETS = [1, 5, 20]   # operating points reported, false alarms per hour
# Onset = first frame inside the label whose level has risen this fraction (in dB) of the way
# from the preceding background to the label's peak. Review-tool labels start a median 150 ms
# before the sound (they inherit the candidate window), which would put the deadline window in
# the silence before the gulp. 0.1 reproduces hand-drawn Audacity starts to a median of 10 ms.
ONSET_RISE = 0.1
# Rise gate: the detector may only fire while the window is at least this far above the
# tracked noise floor. Without it the linear model extrapolates 'quiet = gulp' and digital
# silence (-160 dB at file starts) scores highest of all. Causal; one comparator in hardware.
MIN_RISE_DB = 3.0
BG_S = 0.5               # background level = median over this long before the label
HOP_S = features.HOP / features.FS
FRAME_S = features.WIN / features.FS
OUT = os.path.join(gulpdata.ROOT, 'results')

FEATURE_NAMES = ([f'shape_{i}' for i in range(features.N_BANDS)] +
                 ['level_vs_speech', 'contrast', 'level_std', 'harm_mean', 'harm_max',
                  'flat_mean', 'flux_mean', 'flux_max'])


def n_frames(L_ms):
    """Frames whose total span ((n - 1) hops + one frame) is about L."""
    return max(1, round((L_ms / 1000 - FRAME_S) / HOP_S) + 1)


def mean_n(a, n):
    """Causal mean of the last n values (fewer at the start, never padded)."""
    c = np.concatenate([[0], np.cumsum(a, dtype=np.float64)])
    i = np.arange(len(a))
    lo = np.maximum(0, i - n + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def trailing_features(f, n):
    """Row e = features of frames e-n+1 .. e, i.e. of the audio heard by the end of frame e.

    Also returns a mask of frames where the detector is enabled (no digital silence in the
    floor window, and the window itself not digitally silent).
    """
    bands, total, harm, flat = f['bands'], f['total'], f['harm'], f['flat']
    T = len(total)
    e = np.arange(T)
    s = np.maximum(0, e - n + 1)
    cnt = (e - s + 1)[:, None]

    def mean(a):
        c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0, dtype=np.float64)])
        return (c[e + 1] - c[s]) / (cnt if a.ndim == 2 else cnt[:, 0])

    def tmax(a):
        pad = np.concatenate([np.full(n - 1, -np.inf), a])
        return np.lib.stride_tricks.sliding_window_view(pad, n).max(axis=1)

    mb = mean(bands)
    mt = mean(total)
    flux = np.concatenate([[0], np.maximum(np.diff(bands, axis=0), 0).sum(axis=1)])
    # Noise-floor tracker (minimum statistics): lowest smoothed level in the FLOOR_MS before
    # the window. Unlike a mean, it is not pulled up when speech ends just before a gulp.
    # Causal moving average. (np.convolve would zero-pad, and 0 dB is a loud level, not silence.)
    sm = mean_n(total, FLOOR_SMOOTH)
    fl_n = int(FLOOR_MS / 1000 / HOP_S)
    run_min = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(fl_n, np.inf), sm]), fl_n).min(axis=1)
    floor = run_min[s]            # covers frames s - fl_n .. s - 1
    # Disabled if digital silence occurred within two floor windows: one for the floor window
    # itself and one more so the tracker settles on the room noise (a fade-in after silence
    # otherwise reads as a very low floor).
    dig = (total <= DIGITAL_SILENCE_DB).astype(np.int64)
    c_dig = np.concatenate([[0], np.cumsum(dig)])
    enabled = np.isfinite(floor) & (c_dig[e + 1] - c_dig[np.maximum(0, s - 2 * fl_n)] == 0)
    floor = np.where(enabled, floor, mt)
    p95 = np.percentile(total, 95)
    X = np.column_stack([mb - mb.mean(axis=1, keepdims=True), mt - p95, mt - floor,
                         np.sqrt(np.maximum(mean(total.astype(np.float64) ** 2) - mt ** 2, 0)),
                         mean(harm), tmax(harm), mean(flat), mean(flux), tmax(flux)])
    return X.astype(np.float32), enabled


def frame_end_index(t):
    """Index of the last frame that has fully ended by time t."""
    return int(np.floor((t - FRAME_S) / HOP_S + 1e-9))


def acoustic_onset(total, s, e):
    """Start time of the first frame in [s, e] that rises ONSET_RISE of the way to the peak."""
    a, b = int(round(s / HOP_S)), max(int(round(e / HOP_S)), int(round(s / HOP_S)) + 1)
    bg_n = int(BG_S / HOP_S)
    bg = np.median(total[max(0, a - bg_n):a]) if a > 0 else total[a]
    seg = total[a:b]
    return (a + int(np.argmax(seg >= bg + ONSET_RISE * (seg.max() - bg)))) * HOP_S


def lecture_sets(lec, total, labels, L_s):
    """Events with their decision-frame ranges, and a mask of usable H0 frames."""
    T = len(total)
    end_t = np.arange(T) * HOP_S + FRAME_S
    h0 = np.ones(T, bool)
    hard = np.zeros(T, bool)
    events = []
    for s, e, lab in labels:
        if lab in gulpdata.GULP_LABELS or lab in gulpdata.IGNORE_LABELS:
            h0 &= ~((end_t > s - GUARD_S) & (end_t < e + GUARD_S))
        if lab in gulpdata.GULP_LABELS:
            t0 = acoustic_onset(total, s, e)
            lo, hi = max(0, frame_end_index(t0)), min(T - 1, frame_end_index(t0 + L_s))
            if hi >= lo:
                events.append({'lec': lec, 't0': t0, 'label': lab, 'lo': lo, 'hi': hi})
        elif lab in gulpdata.KEEP_LABELS:
            hard |= (end_t >= s) & (end_t <= e)
    return events, h0, hard & h0


def fa_counts(scores, h0, hold, thresholds):
    """False alarms at each threshold: rising edges of the hold-dilated score in H0 audio."""
    s = np.where(h0, scores, -np.inf)
    pad = np.concatenate([np.full(hold - 1, -np.inf), s])
    d = np.lib.stride_tricks.sliding_window_view(pad, hold).max(axis=1)
    d = np.where(h0, d, -np.inf)            # a firing never 'starts' inside excluded audio
    prev = np.concatenate([[-np.inf], d[:-1]])
    up = d > prev
    lo, hi = np.sort(prev[up]), np.sort(d[up])
    # A rising edge (prev -> d) crosses eta when prev <= eta < d.
    return np.searchsorted(lo, thresholds, 'right') - np.searchsorted(hi, thresholds, 'right')


def run(L_ms, lectures, feats, labels, baseline=False):
    n = n_frames(L_ms)
    L_s = L_ms / 1000
    data = {}
    for lec in lectures:
        X, enabled = trailing_features(feats[lec], n)
        ev, h0, hard = lecture_sets(lec, feats[lec]['total'], labels[lec], L_s)
        data[lec] = (X, ev, h0 & enabled, hard & enabled, enabled)
    folds = [lectures[i::N_FOLDS] for i in range(N_FOLDS)]
    rng = np.random.default_rng(0)
    scored = {}
    for fold in folds:
        if baseline:
            # Energy detector: how far the window rises above the tracked noise floor.
            for lec in fold:
                scored[lec] = data[lec][0][:, FEATURE_NAMES.index('contrast')].astype(np.float64)
            continue
        Xs, ys, ws = [], [], []
        for lec in lectures:
            if lec in fold:
                continue
            X, ev, h0, hard, _ = data[lec]
            for e in ev:
                idx = np.arange((e['lo'] + e['hi'] + 1) // 2, e['hi'] + 1)   # mostly inside the gulp
                Xs.append(X[idx]); ys.append(np.ones(len(idx))); ws.append(np.ones(len(idx)))
            hi_idx = np.flatnonzero(hard)
            Xs.append(X[hi_idx]); ys.append(np.zeros(len(hi_idx))); ws.append(np.full(len(hi_idx), HARD_NEG_WEIGHT))
            pool = np.flatnonzero(h0 & ~hard)
            pick = rng.choice(pool, min(NEG_PER_LECTURE, len(pool)), replace=False)
            Xs.append(X[pick]); ys.append(np.zeros(len(pick))); ws.append(np.ones(len(pick)))
        X, y, w = np.vstack(Xs).astype(np.float64), np.concatenate(ys), np.concatenate(ws)
        w[y == 1] *= w[y == 0].sum() / w[y == 1].sum()
        mu, sd = X.mean(0), X.std(0) + 1e-6
        beta = fit_logreg(np.clip((X - mu) / sd, -Z_CLIP, Z_CLIP), y, w)
        for lec in fold:
            scored[lec] = logit(beta, np.clip((data[lec][0] - mu) / sd, -Z_CLIP, Z_CLIP))
    for lec in lectures:
        rise = data[lec][0][:, FEATURE_NAMES.index('contrast')]
        scored[lec] = np.where((rise >= MIN_RISE_DB) & data[lec][4], scored[lec], -np.inf)

    events, hours = [], 0.0
    for lec in lectures:
        for e in data[lec][1]:
            events.append({**e, 'score': float(scored[lec][e['lo']:e['hi'] + 1].max())})
        hours += data[lec][2].sum() * HOP_S / 3600
    ev_scores = np.array([e['score'] for e in events])
    h0_all = np.concatenate([scored[l][data[l][2]] for l in lectures])
    thresholds = np.unique(np.concatenate([ev_scores, np.quantile(h0_all, np.linspace(0.9, 1, 400))]))
    hold = int(HOLD_MS / 1000 / HOP_S)
    fa = sum(fa_counts(scored[l], data[l][2], hold, thresholds) for l in lectures)
    pd = (ev_scores[None, :] > thresholds[:, None]).mean(axis=1)
    return {'L_ms': L_ms, 'frames': n, 'thresholds': thresholds, 'pd': pd, 'fa_per_h': fa / hours,
            'events': events, 'hours': hours,
            'scored': scored, 'h0': {l: data[l][2] for l in lectures}}   # for export_false_alarms.py


def at_fa(res, target):
    """Best P_D with FA/h <= target, and the threshold that achieves it."""
    ok = res['fa_per_h'] <= target
    if not ok.any():
        return 0.0, np.inf
    i = np.flatnonzero(ok)[np.argmax(res['pd'][ok])]
    return float(res['pd'][i]), float(res['thresholds'][i])


def main():
    lectures = [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]
    feats = {l: features.lecture_features(l) for l in lectures}
    labels = {l: gulpdata.load_labels(l) for l in lectures}

    results = [run(L, lectures, feats, labels) for L in LOOKAHEADS_MS]
    base = run(100, lectures, feats, labels, baseline=True)
    r0 = results[0]
    print(f'{len(r0["events"])} gulps in {len(lectures)} lectures, {r0["hours"]:.1f} h of H0 audio, '
          f'{N_FOLDS}-fold cross-validation by lecture')
    for r in results:
        blocked = sum(1 for e in r['events'] if e['score'] == -np.inf)
        print(f'  rise gate blocks {blocked}/{len(r["events"])} gulps entirely at L = {r["L_ms"]} ms')
    print()
    head = ''.join(f'  P_D @ {t:>2}/h' for t in FA_TARGETS)
    print(f'{"detector":<26}{"window":>8}{head}')
    for r in results + [base]:
        name = f'logistic, L = {r["L_ms"]} ms' if r is not base else 'energy only, L = 100 ms'
        span = (r['frames'] - 1) * HOP_S * 1000 + FRAME_S * 1000
        print(f'{name:<26}{span:>6.0f} ms' + ''.join(f'{at_fa(r, t)[0]:>13.0%}' for t in FA_TARGETS))

    print('\nP_D by severity at 5 false alarms/hour:')
    for r in results:
        _, eta = at_fa(r, 5)
        parts = []
        for lab in ['gulp_low', 'gulp', 'gulp_high', 'gulp_overlap']:
            s = [e['score'] > eta for e in r['events'] if e['label'] == lab]
            if s:
                parts.append(f'{lab} {np.mean(s):.0%} ({sum(s)}/{len(s)})')
        print(f'  L = {r["L_ms"]:>3} ms: ' + ', '.join(parts))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'latency_roc.json'), 'w', encoding='utf-8') as f:
        json.dump({'hours_h0': r0['hours'], 'n_events': len(r0['events']), 'hold_ms': HOLD_MS,
                   'curves': [{'name': (f'L = {r["L_ms"]} ms' if r is not base else 'energy only, L = 100 ms'),
                               'L_ms': r['L_ms'], 'pd': r['pd'].tolist(), 'fa_per_h': r['fa_per_h'].tolist()}
                              for r in results + [base]]}, f)
    print(f'\ncurves saved to {os.path.relpath(os.path.join(OUT, "latency_roc.json"), gulpdata.ROOT)}')


if __name__ == '__main__':
    main()
