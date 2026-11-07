"""Second blind audit: what share of the detector's alarms are gulps further down the scores?

Round 1 (build_audit.py) found 36 of 39 alarms above the measured-5-FA/h threshold to be gulps.
This round uses the labels as they are now (round-1 gulps included), recomputes the
cross-validated scores and the thresholds for measured 5 and 20 FA/h, and samples unlabelled
sounds from two bands:

  A  above the 5 FA/h threshold (whatever is left unlabelled there)   10 sounds
  B  between the 20 and the 5 FA/h thresholds                          30 sounds

plus 6 new labelled-gulp controls and 6 new low-score controls. The items are appended to the
'audit' queue (round-1 items stay, already decided) and the key, with each item's band and the
band sizes, to data/review/audit_key.json.

Usage: .venv\\Scripts\\python tools\\build_audit2.py
"""
import json
import os

import numpy as np

import gulpdata
import onset_classifier as oc
import reanchor as ra
from build_queues import edges_s

SNAP = (0.05, 0.6)
N_A, N_B, N_CTRL = 10, 30, 6


def main():
    keyp = os.path.join(gulpdata.REVIEW, 'audit_key.json')
    qp = os.path.join(gulpdata.REVIEW, 'queue_audit.json')
    K = json.load(open(keyp, encoding='utf-8'))
    Q = json.load(open(qp, encoding='utf-8'))['candidates']
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {}
    for l in lectures:
        D[l] = ra.lecture_data(feats[l], gulpdata.load_labels(l))
        D[l]['total'] = feats[l]['total']
    scores = ra.cv(D, lectures, False)
    sounds = []
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
            lab = any(s - 0.1 <= t <= e + 0.1 for s, e, _ in labels) or f'aud_{l}_{t:.3f}' in K['key']
            sounds.append((best[gi], l, t, lab))
    sc = np.sort([s[0] for s in sounds])
    eta5, eta20 = sc[-int(5 * hours) - 1], sc[-int(20 * hours) - 1]
    A = [s for s in sounds if s[0] > eta5 and not s[3]]
    B = [s for s in sounds if eta20 < s[0] <= eta5 and not s[3]]
    nA = sum(s[0] > eta5 for s in sounds); nB = sum(eta20 < s[0] <= eta5 for s in sounds)
    print(f'{hours:.1f} h, {sum(1 for l in lectures for x in gulpdata.load_labels(l) if x[2] in gulpdata.GULP_LABELS)} gulp labels')
    print(f'band A (> 5 FA/h threshold): {nA} sounds, {len(A)} unlabelled; band B (5-20 FA/h): {nB} sounds, {len(B)} unlabelled')

    rng = np.random.default_rng(7)
    pa = [A[i] for i in rng.choice(len(A), min(N_A, len(A)), replace=False)]
    pb = [B[i] for i in rng.choice(len(B), min(N_B, len(B)), replace=False)]
    low = [s for s in sounds if s[0] < np.percentile(sc, 90) and not s[3]]
    negc = [low[i] for i in rng.choice(len(low), N_CTRL, replace=False)]
    used = {cid for cid in K['key']}
    gl = [(l, s) for l in lectures for s, e, lab in gulpdata.load_labels(l)
          if lab in gulpdata.GULP_LABELS and f'aud_{l}_{s:.3f}' not in used]
    gulpc = [gl[i] for i in rng.choice(len(gl), N_CTRL, replace=False)]

    new = []
    K.setdefault('band', {})
    for kind, band, lst in [('alarm', 'A', [(l, t) for _, l, t, _ in pa]), ('alarm', 'B', [(l, t) for _, l, t, _ in pb]),
                            ('neg_control', None, [(l, t) for _, l, t, _ in negc]), ('gulp_control', None, gulpc)]:
        for l, t in lst:
            cid = f'aud_{l}_{t:.3f}'
            if cid in K['key']:
                continue
            s, e, _ = edges_s(feats[l]['total'], t - SNAP[0], t + SNAP[1])
            new.append({'lec': l, 'start': s, 'end': e, 'onset': round(t, 3), 'score': 0.0, 'id': cid})
            K['key'][cid] = kind
            if band:
                K['band'][cid] = band
    new = [new[i] for i in rng.permutation(len(new))]
    for r, c in enumerate(new, len(Q) + 1):
        c['rank'] = r
    K['round2'] = {'hours': hours, 'eta5': float(eta5), 'eta20': float(eta20), 'nA': int(nA), 'nB': int(nB),
                   'unlabelled_A': len(A), 'unlabelled_B': len(B)}
    with open(qp, 'w', encoding='utf-8') as f:
        json.dump({'candidates': Q + new}, f, indent=1)
    with open(keyp, 'w', encoding='utf-8') as f:
        json.dump(K, f, indent=1)
    print(f'appended {len(new)} items to the audit queue (now {len(Q) + len(new)})')


if __name__ == '__main__':
    main()
