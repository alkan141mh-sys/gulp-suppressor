"""Catch rate against CORRECTED false alarms, using the blind audits.

The measured false-alarm count treats every unlabelled sound as a non-gulp; the audits showed
most high-scoring ones are gulps. Here the detector is retrained (same CV) on today's labels,
the non-gulp sounds are ranked by score, and each unlabelled one counts as a false alarm only
with probability 1 - p, where p is the audited gulp share at that depth:
  top ~5 FA/h worth of sounds        p = 0.90  (rounds 1+2 band A: 44 of 49)
  next, down to ~20 FA/h             p = 0.50  (round 2 band B: 15 of 30)
  below                              p = 0     (not audited; counted as false alarms: conservative)
Sounds with a non-gulp label always count. Catch rate is over the labelled gulps; those now
include gulps found BY the detector, so it is optimistic for the gulps nobody has found yet.

Usage: .venv\\Scripts\\python tools\\corrected_roc.py
"""
import numpy as np

import gulpdata
import onset_classifier as oc
import reanchor as ra

P_TOP, P_MID = 0.90, 0.50
VETO_DB = None
VETO_FRAMES = 8         # window for the veto: 8 frames = the decision window (live mode)
BREATH = False


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {}
    for l in lectures:
        D[l] = ra.lecture_data(feats[l], gulpdata.load_labels(l))
        D[l]['total'] = feats[l]['total']
    n_g = sum(len(d['gulps']) for d in D.values())
    scores = ra.cv(D, lectures, False)
    if VETO_DB is not None:
        # Level veto: nothing in the decision window louder than any labelled gulp (max 40 dB
        # above the floor) can be a gulp; a linear model extrapolates past that.
        for l in lectures:
            tot = D[l]['total']
            _, floor = oc.detect_onsets(tot)
            A = D[l]['A']
            idx = np.clip(A[:, None] + np.arange(VETO_FRAMES)[None, :], 0, len(tot) - 1)
            pk = tot[idx].max(1) - floor[A]
            scores[l] = np.where(pk > VETO_DB, -np.inf, scores[l])
            D[l]['vetoed'] = int((pk > VETO_DB).sum())
        print(f'veto at {VETO_DB} dB above the floor: {sum(D[l]["vetoed"] for l in lectures)} anchors vetoed')
    if BREATH:
        # The user's breath rule (tools/breath_rule.py): unvoiced, below 35 dB, steady low band
        # (mean |change| < 2 dB per 10 ms) and either hissy (tilt > -18 dB) or weak below
        # 3.2 kHz (< 16 dB). Decided per sound, from its onset.
        from breath_rule import sound_feats
        nb = 0
        for l in lectures:
            tot = D[l]['total']
            on, floor = oc.detect_onsets(tot)
            f = feats[l]
            veto = np.zeros(len(on), bool)
            for gi, k0 in enumerate(on):
                x = sound_feats(f, floor, int(k0))
                veto[gi] = x['gate'] and x['lf_rough'] < 2.0 and (x['tilt'] > -18 or x['lf_peak'] < 16)
            scores[l] = np.where(veto[D[l]['grp']], -np.inf, scores[l])
            nb += veto.sum()
            D[l]['breath_veto'] = veto
        print(f'breath rule: {nb} sounds vetoed')
    neg, pos = [], []
    for l in lectures:
        d, z = D[l], scores[l]
        on, _ = oc.detect_onsets(d['total'])
        labels = gulpdata.load_labels(l)
        bad = np.zeros(d['grp'].max() + 1, bool)
        bad[d['grp'][d['y'] != 0]] = True
        best = np.full(len(bad), -np.inf)
        ok = d['y'] == 0
        np.maximum.at(best, d['grp'][ok], z[ok])
        for gi in np.flatnonzero(~bad & np.isfinite(best)):
            t = on[gi] * oc.HOP_S
            neg.append((best[gi], any(s - 0.1 <= t <= e + 0.1 for s, e, _ in labels)))
        for g in d['gulps']:
            a = g['anchors']
            pos.append(z[a].max() if len(a) else -np.inf)
    neg.sort(key=lambda x: -x[0])
    pos = np.array(pos)
    n5, n20 = int(5 * hours), int(20 * hours)
    w = np.array([1.0 if lab else (1 - P_TOP if r < n5 else 1 - P_MID if r < n20 else 1.0)
                  for r, (_, lab) in enumerate(neg)])
    s = np.array([x[0] for x in neg])
    cum = np.cumsum(w) / hours            # corrected FA/h if the threshold sits just below rank r
    print(f'{hours:.1f} h, {n_g} labelled gulps, retrained in 6-fold lecture-grouped CV\n')
    print(f'{"true FA/h":>10}{"measured FA/h":>15}{"caught":>9}')
    for target in (0.5, 1, 2, 3, 5, 10):
        r = np.searchsorted(cum, target, side='right')         # alarms allowed
        eta = s[r] if r < len(s) else -np.inf
        print(f'{target:>10}{r / hours:>15.1f}{np.mean(pos > eta):>9.0%}')
    print(f'\n(one gulp = {100 / n_g:.1f} points)')


if __name__ == '__main__':
    import sys
    if '--veto' in sys.argv:
        VETO_DB = float(sys.argv[sys.argv.index('--veto') + 1])
    BREATH = '--breath' in sys.argv
    if '--veto-frames' in sys.argv:
        VETO_FRAMES = int(sys.argv[sys.argv.index('--veto-frames') + 1])
    main()
