"""Deal with unlabelled gulps hiding among the negative onsets (positive-unlabelled learning).

Most training lectures were never listened to end to end, so some 'other onsets' are gulps.
When the user listened to the onset classifier's 52 strongest false alarms, 75% were gulps.

1. Calibration. Reproduce the exact model that produced that list (N = 100 ms, lambda 100,
   labels as they were before those 52 decisions) and measure the gulp share among the
   reviewed items. A fitted P(gulp | score) curve is not usable: inside the reviewed range the
   share falls slightly with score, and extrapolating that downwards calls almost every speech
   onset a gulp. So the share is taken as flat, and only inside the listed score range.
2. Every negative onset nobody listened to with a score inside that range gets p = the
   reviewed gulp share; everything below it, and every reviewed onset, gets p = 0. 'Measured'
   (p = 0 everywhere) and 'corrected' then bracket the truth for the listed score range.
3. Training: negatives are weighted by (1 - p), so a likely gulp barely counts as 'not gulp'.
4. Metrics, corrected: expected false alarms = sum of (1 - p) over negatives above the
   threshold; expected unlabelled gulps (sum of p) join the gulp count, detected ones too.

Limits: the calibration rests on 52 items with scores between about 1 and 8; below that it is
extrapolated. Gulps that score low under every model stay invisible (the miss blind spot).

Usage: .venv\\Scripts\\python tools\\pu_correction.py
"""
import json
import os

import numpy as np

import gulpdata
import onset_classifier as oc
from find_candidates import fit_logreg, logit

REVIEW_PREFIX = 'on100_'      # decisions made on candidates_onset.json
N_LIST = [50, 75, 100]
LAM = 100.0


def onset_index(on, t):
    return int(np.argmin(np.abs(on * oc.HOP_S - t)))


def reviewed_mask(on, labels):
    t = on * oc.HOP_S
    m = np.zeros(len(on), bool)
    for s, e, _ in labels:
        m |= (t >= s - 0.05) & (t <= e)
    return m


def metrics(scores, base, p, hours, targets, corrected):
    pos, neg, pneg = [], [], []
    for l, (on, floor, y, w, gulps) in base.items():
        neg.append(scores[l][y == 0])
        pneg.append(p[l][y == 0])
        for g in gulps:
            pos.append(scores[l][g['onset_index']] if g['onset_index'] is not None else -np.inf)
    pos, neg, pneg = np.array(pos), np.concatenate(neg), np.concatenate(pneg)
    o = np.argsort(-neg)
    neg, pneg = neg[o], pneg[o]
    fa_cost = np.cumsum(1 - pneg) if corrected else np.arange(1, len(neg) + 1, dtype=float)
    extra = np.cumsum(pneg) if corrected else np.zeros(len(neg))
    n_hidden = pneg.sum() if corrected else 0.0
    out = []
    for t in targets:
        k = np.searchsorted(fa_cost, t * hours, 'right')       # negatives allowed above eta
        eta = neg[k] if k < len(neg) else -np.inf
        found = (pos > eta).sum() + (extra[k - 1] if k > 0 else 0.0)
        out.append(found / (len(pos) + n_hidden))
    return out


def main():
    # 1. The model exactly as it was when the onset false-alarm list was made.
    lectures, feats, base0 = oc.load_all(exclude=lambda cid: cid.startswith(REVIEW_PREFIX))
    s0 = oc.cv_scores(100, LAM, lectures, feats, base0)
    with open(os.path.join(gulpdata.REVIEW, 'candidates_onset.json'), encoding='utf-8') as f:
        listed = json.load(f)['candidates']
    dec = gulpdata.load_decisions()
    xs, ys, drift = [], [], []
    for c in listed:
        d = dec.get(c['id'])
        if not d or d['label'] in gulpdata.IGNORE_LABELS:
            continue
        on = base0[c['lec']][0]
        i = onset_index(on, c['start'] + 0.05)
        xs.append(s0[c['lec']][i])
        ys.append(float(d['label'] in gulpdata.GULP_LABELS))
        drift.append(abs(s0[c['lec']][i] - c['score']))
    xs, ys = np.array(xs), np.array(ys)
    print(f'reproduced the listing model: max score difference {max(drift):.3f} on {len(xs)} reviewed items')
    share = ys.mean()
    half = np.median(xs)
    print(f'  reviewed: {int(ys.sum())}/{len(ys)} gulps ({share:.0%}), scores {xs.min():.1f} to {xs.max():.1f}; '
          f'gulp share {ys[xs < half].mean():.0%} in the lower half, {ys[xs >= half].mean():.0%} in the upper half')
    range_lo = min(c['score'] for c in listed)          # lowest score on the whole listing
    P = lambda s: np.where(s >= range_lo, share, 0.0)
    print(f'  assumed: unreviewed onsets scoring >= {range_lo:.1f} are gulps with p = {share:.2f}, below: p = 0')

    # 2. Gulp probability for every negative onset of the current label set.
    lectures, feats, base = oc.load_all()
    p, w_pu = {}, {}
    hidden_in_range = hidden_below = 0.0
    for l in lectures:
        on, floor, y, w, gulps = base[l]
        pl = np.zeros(len(on))
        unrev = (y == 0) & ~reviewed_mask(on, gulpdata.load_labels(l))
        pl[unrev] = P(s0[l][unrev])
        hidden_in_range += pl[unrev & (s0[l] >= xs.min())].sum()
        hidden_below += pl[unrev & (s0[l] < xs.min())].sum()
        p[l] = pl
        w_pu[l] = np.where(y == 0, w * (1 - pl), w)
    n_lab = sum(len(b[4]) for b in base.values())
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    n_unrev = sum(int((p[l] > 0).sum()) for l in lectures)
    print(f'\nunreviewed onsets in the listed score range: {n_unrev}; expected gulps among them: '
          f'{hidden_in_range + hidden_below:.0f} (labelled gulps: {n_lab})\n')

    # 3. Train plain vs PU-weighted, report measured vs corrected metrics.
    base_pu = {l: (b[0], b[1], b[2], w_pu[l], b[4]) for l, b in base.items()}
    print(f'{"N":<7}{"training":<11}{"metric":<11}' + ''.join(f'{"P_D @ " + str(t) + "/h":>13}' for t in oc.FA_TARGETS))
    for N in N_LIST:
        for name, b in [('plain', base), ('PU', base_pu)]:
            sc = oc.cv_scores(N, LAM, lectures, feats, b)
            for corr in [False, True]:
                row = metrics(sc, base, p, hours, oc.FA_TARGETS, corr)
                print(f'{N:>3} ms {name:<11}{"corrected" if corr else "measured":<11}' +
                      ''.join(f'{v:>13.0%}' for v in row), flush=True)


if __name__ == '__main__':
    main()
