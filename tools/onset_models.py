"""Compare classifiers for stage 2 of the onset detector (onset_classifier.py), same data and CV.

  logistic       the linear baseline (lambda 10)
  boosted_trees  gradient-boosted decision trees: nonlinear, little tuning, handles
                 interactions such as 'rises fast AND has no harmonics'
  mlp            one hidden layer of 32 units: nonlinear and cheap in fixed point
                 (one multiply-accumulate per weight), so the FPGA-friendly candidate

Training-only augmentation: each gulp onset is also used one frame (10 ms) earlier and later,
because the onset detector's frame will not always match. Hyperparameters are fixed in advance,
not tuned on these results.

Usage: .venv\\Scripts\\python tools\\onset_models.py
"""
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

import onset_classifier as oc
from find_candidates import fit_logreg, logit
from output_roc import n_future

LOOKAHEADS_MS = [50, 75, 100]
MODELS = ['logistic', 'boosted_trees', 'mlp']
SHIFTS = [-1, 1]


def train_predict(name, X, y, w, Xte):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z, Zte = np.clip((X - mu) / sd, -4, 4), np.clip((Xte - mu) / sd, -4, 4)
    if name == 'logistic':
        return logit(fit_logreg(Z, y, w, lam=10.0), Zte)
    if name == 'boosted_trees':
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=20, l2_regularization=1.0,
                                           early_stopping=False, random_state=0)
        return m.fit(X, y, sample_weight=w).decision_function(Xte)
    if name == 'mlp':
        m = MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=150, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            m.fit(Z, y, sample_weight=w / w.mean())
        p = np.clip(m.predict_proba(Zte)[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))
    raise ValueError(name)


def main():
    lectures, feats, base = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    folds = [lectures[i::oc.N_FOLDS] for i in range(oc.N_FOLDS)]
    n_gulps = sum(len(b[4]) for b in base.values())
    print(f'{hours:.1f} h, {n_gulps} gulps, {sum((b[2] == 0).sum() for b in base.values())} other onsets\n')
    print(f'{"N":<8}{"model":<15}' + ''.join(f'{"P_D @ " + str(t) + "/h":>13}' for t in oc.FA_TARGETS))
    for N in LOOKAHEADS_MS:
        n_after = n_future(N)
        X = {l: oc.patches(feats[l], base[l][0], base[l][1], n_after) for l in lectures}
        aug = {}
        for l in lectures:
            on, floor, y, w, _ = base[l]
            pos = on[y == 1]
            T = len(feats[l]['total'])
            aug[l] = np.vstack([oc.patches(feats[l], np.clip(pos + s, 0, T - 1), floor, n_after) for s in SHIFTS]) \
                if len(pos) else np.empty((0, X[l].shape[1]))
        for name in MODELS:
            scores = {}
            for fold in folds:
                tr = [l for l in lectures if l not in fold]
                Xt = np.vstack([X[l][base[l][2] >= 0] for l in tr] + [aug[l] for l in tr])
                yt = np.concatenate([base[l][2][base[l][2] >= 0] for l in tr] + [np.ones(len(aug[l])) for l in tr]).astype(float)
                wt = np.concatenate([base[l][3][base[l][2] >= 0] for l in tr] + [np.ones(len(aug[l])) for l in tr])
                wt[yt == 1] *= wt[yt == 0].sum() / wt[yt == 1].sum()
                Xte = np.vstack([X[l] for l in fold])
                z = train_predict(name, Xt, yt, wt, Xte)
                i = 0
                for l in fold:
                    scores[l] = z[i:i + len(X[l])]
                    i += len(X[l])
            pos, neg, labs = [], [], []
            for l in lectures:
                on, floor, y, w, gulps = base[l]
                neg.append(scores[l][y == 0])
                for g in gulps:
                    pos.append(scores[l][g['onset_index']] if g['onset_index'] is not None else -np.inf)
                    labs.append(g['label'])
            pos, neg = np.array(pos), np.sort(np.concatenate(neg))
            row = []
            for t in oc.FA_TARGETS:
                k = int(t * hours)
                eta = neg[-k - 1] if k < len(neg) else -np.inf
                row.append((pos > eta).mean())
            eta5 = neg[-int(5 * hours) - 1]
            sev = ', '.join(f'{lab} {np.mean([p > eta5 for p, q in zip(pos, labs) if q == lab]):.0%}'
                            for lab in ['gulp_low', 'gulp', 'gulp_high'])
            print(f'{N:>3} ms   {name:<15}' + ''.join(f'{v:>13.0%}' for v in row) + f'    @5/h: {sev}', flush=True)


if __name__ == '__main__':
    main()
