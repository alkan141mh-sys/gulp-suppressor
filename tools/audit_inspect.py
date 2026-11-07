"""Inspect the audit items: for each, its hidden kind/band, the user's label, and simple
measurements in the 100 ms after the onset (peak level above the floor, peak harmonicity,
voiced frames), to see whether the detector's non-gulp alarms are obviously speech.

Usage: .venv\\Scripts\\python tools\\audit_inspect.py
"""
import json
import os

import numpy as np

import features
import gulpdata as g
import onset_classifier as oc


def main():
    K = json.load(open(os.path.join(g.REVIEW, 'audit_key.json'), encoding='utf-8'))
    d = g.load_decisions()
    cache = {}
    rows = []
    for cid, kind in K['key'].items():
        _, lec, t = cid.split('_')
        t = float(t)
        if lec not in cache:
            f = features.lecture_features(lec)
            cache[lec] = (f, oc.detect_onsets(f['total'])[1])
        f, floor = cache[lec]
        k = int(round(t / oc.HOP_S))
        seg = slice(k, k + 10)
        lev = f['total'][seg] - floor[k]
        h = f['harm'][seg]
        rows.append((kind, K.get('band', {}).get(cid, '1' if kind == 'alarm' else '-'), d.get(cid, {}).get('label'),
                     lec, t, lev.max(), h.max(), int(((h >= 0.8) & (lev >= 25)).sum())))
    rows.sort(key=lambda r: (r[0], r[2] in g.GULP_LABELS, -r[5]))
    print(f'{"kind":<13}{"band":<5}{"label":<10}{"lec":<5}{"time":>9}{"peak dB":>9}{"harm":>6}{"voiced":>7}')
    for r in rows:
        m, s = divmod(r[4], 60)
        print(f'{r[0]:<13}{r[1]:<5}{str(r[2]):<10}{r[3]:<5}{f"{int(m)}:{s:06.3f}":>9}{r[5]:>9.1f}{r[6]:>6.2f}{r[7]:>7}')


if __name__ == '__main__':
    main()
