"""Learning curve: how does the detector improve with the number of labelled gulps?

The current detector (reanchor.py: 100 ms onset patch + re-anchoring, logistic regression) is
trained, in the usual lecture-grouped 6-fold cross-validation, on only a random fraction of the
training gulps (the others are left out of training, not turned into negatives). Testing
always uses every gulp. If the curve is still rising at 100%, more gulps would help; if it is
flat, the limit is elsewhere (features, model, label noise).

Usage: .venv\\Scripts\\python tools\\learning_curve.py
"""
import copy

import numpy as np

import gulpdata
import onset_classifier as oc
import reanchor as ra

FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SEEDS = 8


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {}
    for l in lectures:
        D[l] = ra.lecture_data(feats[l], gulpdata.load_labels(l))
        D[l]['total'] = feats[l]['total']
    n = sum(len(d['gulps']) for d in D.values())
    print(f'{n} gulps, {hours:.1f} h; caught share at each FA/h, mean (min-max) over {SEEDS} random subsets\n')
    print(f'{"gulps used in training":<26}' + ''.join(f'{"@ " + str(t) + " FA/h":>20}' for t in oc.FA_TARGETS))
    for frac in FRACTIONS:
        rows = []
        for seed in range(SEEDS if frac < 1 else 1):
            rng = np.random.default_rng(seed)
            Dt = {}
            for l in lectures:
                d = copy.copy(D[l])
                y = d['y'].copy()
                for g in d['gulps']:
                    if rng.random() >= frac:
                        a = g['anchors']
                        y[a[y[a] == 1]] = -1          # this gulp is not used for training
                d['y'] = y
                Dt[l] = d
            scores = ra.cv(Dt, lectures, False)
            rows.append([a for a, _, _ in ra.evaluate(D, scores, lectures, hours, oc.FA_TARGETS)])
        R = np.array(rows)
        cells = ''.join(f'{f"{R[:, j].mean():.0%} ({R[:, j].min():.0%}-{R[:, j].max():.0%})":>20}' if len(R) > 1
                        else f'{R[0, j]:>20.0%}' for j in range(R.shape[1]))
        print(f'{f"{frac:.0%} (~{round(frac * n * 5 / 6)} per fold)":<26}' + cells, flush=True)


if __name__ == '__main__':
    main()
