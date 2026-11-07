"""The user's cues for the detector's false alarms, measured on gulps and on those alarms.

Cues (from the user's notes on the 20 confirmed false alarms):
  long        the sound goes on far longer than a gulp
  speech      it turns into loud, voiced, high-frequency speech
  breath      it starts as a steady hiss with no burst (stationary spectrum, no transient)

For each labelled gulp (at its detected onset/anchor) and each false alarm (at its onset), over
the W ms after the onset: peak level above the floor, voiced frames (harmonicity >= 0.8 at
>= floor + V dB), the sound's duration (until 100 ms back near the floor, capped at 1.5 s) and
a burstiness measure (max frame-to-frame level rise in the first 100 ms).

Usage: .venv\\Scripts\\python tools\\cue_check.py
"""
import json
import os

import numpy as np

import features
import gulpdata as g
import onset_classifier as oc
import reanchor as ra


def measures(f, floor, k0):
    tot, harm = f['total'], f['harm']
    T = len(tot)
    fl = floor[k0]
    out = {}
    for W in (20, 30, 40):            # frames
        seg = slice(k0, min(T, k0 + W))
        lev = tot[seg] - fl
        out[f'peak{W}'] = lev.max()
        for V in (15, 25):
            out[f'voiced{W}_{V}'] = int(((harm[seg] >= 0.8) & (lev >= V)).sum())
    quiet, end = 0, min(T, k0 + 150)
    for k in range(k0 + 1, min(T, k0 + 150)):
        if tot[k] < fl + oc.ONSET_DB:
            quiet += 1
            if quiet >= 10:
                end = k - 9
                break
        else:
            quiet = 0
    out['dur'] = (end - k0) * 10
    out['rise'] = np.diff(tot[k0:k0 + 10]).max()
    return out


def main():
    K = json.load(open(os.path.join(g.REVIEW, 'audit_key.json'), encoding='utf-8'))
    d = g.load_decisions()
    lectures = [l for l in g.lectures() if l not in g.TEST_LECTURES]
    G, FA = [], []
    for l in lectures:
        f = features.lecture_features(l)
        on, floor = oc.detect_onsets(f['total'])
        D = ra.lecture_data(f, g.load_labels(l))
        for gg in D['gulps']:
            a = gg['anchors']
            if len(a):
                G.append(measures(f, floor, int(D['A'][a[0]])))
        for cid, kind in K['key'].items():
            if kind == 'alarm' and cid.split('_')[1] == l:
                lab = d.get(cid, {}).get('label')
                if lab and lab not in g.GULP_LABELS and lab != 'unsure':
                    FA.append(measures(f, floor, int(round(float(cid.split('_')[2]) / oc.HOP_S))))
    print(f'{len(G)} gulps (at their first anchor), {len(FA)} confirmed false alarms\n')
    keys = list(G[0])
    print(f'{"measure":<14}{"gulps p50/p95/p99/max":>30}{"false alarms (sorted)":>8}')
    for k in keys:
        a = np.array([x[k] for x in G]); b = np.sort([x[k] for x in FA])
        print(f'{k:<14}{f"{np.median(a):.0f} / {np.percentile(a, 95):.0f} / {np.percentile(a, 99):.0f} / {a.max():.0f}":>30}   '
              + ' '.join(f'{v:.0f}' for v in b))


if __name__ == '__main__':
    main()
