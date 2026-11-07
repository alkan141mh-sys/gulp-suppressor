"""Blind audit of the detector's false alarms: how many of them are really gulps?

Every model tried stops at about 69% caught at 5 FA/h. If a large share of those "false alarms"
are unlabelled gulps, the ceiling is in the labels, not in the models. This builds a blind
review queue (source 'audit' in the review tool):

  alarms     sounds the current detector (reanchor.py, cross-validated) fires on at 5 FA/h that
             have no label of any kind; up to 40 of them, chosen at random
  controls   6 labelled gulps and 6 sounds the detector scores well below the threshold,
             so the blind judgement itself can be checked (would you call a known gulp a gulp?)

All are shuffled, shown without score, with edges measured from the audio. Which item is which
is written to data/review/audit_key.json, which the review tool never serves.

Usage: .venv\\Scripts\\python tools\\build_audit.py
"""
import json
import os

import numpy as np

import gulpdata
import onset_classifier as oc
import reanchor as ra
from build_queues import edges_s

FA_PER_H = 5
N_ALARMS, N_GULP_CTRL, N_NEG_CTRL = 40, 6, 6
SNAP = (0.05, 0.6)


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {}
    for l in lectures:
        D[l] = ra.lecture_data(feats[l], gulpdata.load_labels(l))
        D[l]['total'] = feats[l]['total']
    scores = ra.cv(D, lectures, False)

    sounds = []          # (score, lecture, onset time, labelled?) for sounds touching no gulp/ignore span
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
            lab = any(s - 0.1 <= t <= e + 0.1 for s, e, _ in labels)
            sounds.append((best[gi], l, t, lab))
    sc = np.sort([s[0] for s in sounds])
    eta = sc[-int(FA_PER_H * hours) - 1]
    alarms = [s for s in sounds if s[0] > eta]
    unlab = [s for s in alarms if not s[3]]
    print(f'{hours:.1f} h; threshold for {FA_PER_H} FA/h: {len(alarms)} false alarms, '
          f'{len(alarms) - len(unlab)} of them already labelled (not_gulp, breath, ...), {len(unlab)} unlabelled')

    rng = np.random.default_rng(2026)
    pick = [unlab[i] for i in rng.choice(len(unlab), min(N_ALARMS, len(unlab)), replace=False)]
    low = [s for s in sounds if s[0] < np.percentile(sc, 90) and not s[3]]
    negc = [low[i] for i in rng.choice(len(low), N_NEG_CTRL, replace=False)]
    gl = [(l, s, e) for l in lectures for s, e, lab in gulpdata.load_labels(l) if lab in gulpdata.GULP_LABELS]
    gulpc = [gl[i] for i in rng.choice(len(gl), N_GULP_CTRL, replace=False)]

    items, key = [], {}
    for kind, lst in [('alarm', [(l, t) for _, l, t, _ in pick]), ('neg_control', [(l, t) for _, l, t, _ in negc]),
                      ('gulp_control', [(l, s) for l, s, _ in gulpc])]:
        for l, t in lst:
            s, e, _ = edges_s(feats[l]['total'], t - SNAP[0], t + SNAP[1])
            cid = f'aud_{l}_{t:.3f}'
            items.append({'lec': l, 'start': s, 'end': e, 'onset': round(t, 3), 'score': 0.0, 'id': cid})
            key[cid] = kind
    items = [items[i] for i in rng.permutation(len(items))]
    for r, c in enumerate(items, 1):
        c['rank'] = r
    with open(os.path.join(gulpdata.REVIEW, 'queue_audit.json'), 'w', encoding='utf-8') as f:
        json.dump({'candidates': items}, f, indent=1)
    with open(os.path.join(gulpdata.REVIEW, 'audit_key.json'), 'w', encoding='utf-8') as f:
        json.dump({'eta': float(eta), 'n_alarms': len(alarms), 'n_unlabelled': len(unlab), 'hours': hours,
                   'key': key}, f, indent=1)
    print(f'audit queue: {len(items)} items ({len(pick)} alarms, {len(negc)} + {len(gulpc)} controls), shuffled')


if __name__ == '__main__':
    main()
