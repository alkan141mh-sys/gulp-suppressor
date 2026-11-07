"""Put the onset classifier's strongest false alarms into the review tool.

Same idea as export_false_alarms.py, for the two-stage detector of onset_classifier.py:
the highest-scoring onsets that are not labelled gulps, scored by the cross-validated model
(no lecture scored by a model trained on it). Onsets already labelled, decided or queued
in another candidate set are skipped. Writes data/review/candidates_onset.json.

Usage: .venv\\Scripts\\python tools\\export_onset_false_alarms.py [--count 150 --N 100 --lam 100]
"""
import argparse
import json
import os

import numpy as np

import gulpdata
import onset_classifier as oc

SPAN_BEFORE_S = 0.05    # review span around the onset: a little before ...
SPAN_AFTER_S = 0.30     # ... and enough after to hear what the sound is


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--lam', type=float, default=100.0)
    args = ap.parse_args()

    lectures, feats, base = oc.load_all()
    scores = oc.cv_scores(args.N, args.lam, lectures, feats, base)

    known = {l: [(s, e) for s, e, _ in gulpdata.load_labels(l)] for l in lectures}
    for name in ['candidates.json', 'candidates_fa.json']:
        p = os.path.join(gulpdata.REVIEW, name)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                for c in json.load(f)['candidates']:
                    if c['lec'] in known:
                        known[c['lec']].append((c['start'], c['end']))

    out = []
    for l in lectures:
        on, floor, y, w, gulps = base[l]
        for i in np.flatnonzero(y == 0):
            t = on[i] * oc.HOP_S
            t0, t1 = max(0.0, t - SPAN_BEFORE_S), t + SPAN_AFTER_S
            if any(t0 < b and a < t1 for a, b in known[l]):
                continue
            out.append({'lec': l, 'start': round(t0, 3), 'end': round(t1, 3), 'score': round(float(scores[l][i]), 2)})
    out.sort(key=lambda c: -c['score'])
    out = out[:args.count]
    for r, c in enumerate(out, 1):
        c['id'] = f"on{args.N}_{c['lec']}_{c['start']:.3f}"
        c['rank'] = r

    path = os.path.join(gulpdata.REVIEW, 'candidates_onset.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'model': {'N_ms': args.N, 'lambda': args.lam}, 'candidates': out}, f, indent=1)
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    print(f'{len(out)} onset false alarms written to {os.path.relpath(path, gulpdata.ROOT)} '
          f'(the strongest {len(out) / hours:.1f}/h; score range {out[-1]["score"]:.1f}-{out[0]["score"]:.1f})')


if __name__ == '__main__':
    main()
