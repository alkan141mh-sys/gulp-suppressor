"""Whole-sound classification: with read-ahead the detector can see the ENTIRE sound and what
comes after it before deciding, not just the first 100 ms.

Same detection units as reanchor.py (an onset plus up to 3 secondary anchors = one sound; a
gulp is caught when any anchor inside it scores above the threshold). Each anchor keeps its
100 ms patch (group A) and gets features of the whole sound and its surroundings:

  B  sound shape    duration (to 100 ms back near the floor, capped at 1.5 s), whether it hit
                    the cap, peak level and time to peak, the sound's mean spectrum shape
                    (24 bands relative to their maximum), voicing (share of voiced frames, max
                    and mean harmonicity), number of secondary anchors, anchor position in it
  C  before         time since the last voiced speech, loudest level in the 2 s before
  D  after          time from the sound's end to the next voiced speech, and its level

Voiced speech = harmonicity >= 0.8 and level >= floor + 25 dB (the mute rule's definition).
All of it is available ahead of playback in read-ahead mode (up to 1.5 s + 3 s after onset).

Models, all with the same lecture-grouped 6-fold cross-validation:
  A logistic          the current detector (reanchor.py)
  A+B+C+D logistic    same model, more features
  B+C+D logistic      compact: sound-level features only
  B+C+D boosted       gradient-boosted trees (non-linear), sound-level features
  A logistic + boosted  average of the two log-odds

Usage: .venv\\Scripts\\python tools\\whole_sound.py
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import onset_classifier as oc
import reanchor as ra
import gulpdata
from find_candidates import fit_logreg, logit

MAX_FR = 150            # sound length cap, frames (1.5 s)
END_QUIET = 10
SPEECH_HARM, SPEECH_DB = 0.8, 25.0
CTX_BEFORE = 500        # frames (5 s) searched back for speech
CTX_AFTER = 300         # frames (3 s) searched ahead


def sound_features(f, D):
    total, harm, bands = f['total'], f['harm'], f['bands']
    T = len(total)
    on, floor = oc.detect_onsets(total)
    voiced = np.flatnonzero((harm >= SPEECH_HARM) & (total >= floor + SPEECH_DB))
    A, grp, kind = D['A'], D['grp'], D['kind']
    n2 = np.bincount(grp, weights=kind, minlength=len(on))
    per_sound = []
    for gi, k0 in enumerate(on):
        fl = floor[k0]
        lim = min(T, k0 + MAX_FR)
        quiet, end, capped = 0, None, 0
        for k in range(k0 + 1, lim):
            if total[k] < fl + oc.ONSET_DB:
                quiet += 1
                if quiet >= END_QUIET:
                    end = k - END_QUIET + 1
                    break
            else:
                quiet = 0
        if end is None:
            end, capped = lim, 1
        seg = slice(k0, max(end, k0 + 1))
        lev = total[seg] - fl
        pk = int(np.argmax(lev))
        P = np.mean(10 ** (bands[seg] / 10), axis=0)
        prof = 10 * np.log10(P + 1e-12)
        prof -= prof.max()
        h = harm[seg]
        w = 10 ** (lev / 10)
        vfrac = np.mean((h >= SPEECH_HARM) & (lev >= SPEECH_DB))
        # context
        i = np.searchsorted(voiced, k0) - 1
        since = (k0 - voiced[i]) if i >= 0 and k0 - voiced[i] <= CTX_BEFORE else CTX_BEFORE
        before_max = (total[max(0, k0 - 200):k0].max() - fl) if k0 > 0 else 0.0
        j = np.searchsorted(voiced, end)
        until = (voiced[j] - end) if j < len(voiced) and voiced[j] - end <= CTX_AFTER else CTX_AFTER
        after_lev = (total[voiced[j]:voiced[j] + 20].max() - fl) if until < CTX_AFTER else 0.0
        per_sound.append([
            (end - k0) * 10, capped, lev.max(), pk * 10, *prof, vfrac, h.max(), np.sum(h * w) / np.sum(w),
            n2[gi], since * 10, before_max, until * 10, after_lev, float(until < 30),
        ])
    S = np.array(per_sound)
    # E: the 100 ms patch around the sound's loudest frame (its main burst), wherever it falls
    peaks = np.array([k0 + int(s[3] // 10) for k0, s in zip(on, per_sound)])
    E = oc.patches(f, np.clip(peaks - 3, 0, T - 1), floor, 13)        # 30 ms before .. 130 ms after the peak - 3
    D['XE'] = E[grp]
    # anchor position inside its sound
    dur = np.maximum(S[grp, 0], 10)
    pos = (A - on[grp]) * 10 / dur
    return np.hstack([S[grp], pos[:, None]])


def weights(D, l, sel):
    return D[l]['w'][sel]


def cv_scores(D, lectures, key, model):
    scores = {}
    for fold in [lectures[i::oc.N_FOLDS] for i in range(oc.N_FOLDS)]:
        tr = [l for l in lectures if l not in fold]
        sel = {l: D[l]['y'] >= 0 for l in tr}
        Xt = np.vstack([D[l][key][sel[l]] for l in tr])
        yt = np.concatenate([D[l]['y'][sel[l]] for l in tr]).astype(float)
        wt = np.concatenate([D[l]['w'][sel[l]] for l in tr])
        wt[yt == 1] *= wt[yt == 0].sum() / wt[yt == 1].sum()
        if model == 'logistic':
            mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
            beta = fit_logreg(np.clip((Xt - mu) / sd, -4, 4), yt, wt, lam=ra.LAM)
            for l in fold:
                scores[l] = logit(beta, np.clip((D[l][key] - mu) / sd, -4, 4))
        else:
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                 min_samples_leaf=40, l2_regularization=1.0, random_state=0)
            clf.fit(Xt, yt, sample_weight=wt)
            for l in fold:
                p = np.clip(clf.predict_proba(D[l][key])[:, 1], 1e-9, 1 - 1e-9)
                scores[l] = np.log(p / (1 - p))
    return scores


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {}
    for l in lectures:
        d = ra.lecture_data(feats[l], gulpdata.load_labels(l))
        d['total'] = feats[l]['total']
        d['XS'] = sound_features(feats[l], d)
        d['XA'] = d['X']
        d['XAS'] = np.hstack([d['X'], d['XS']])
        d['XAE'] = np.hstack([d['X'], d['XE']])
        d['XAES'] = np.hstack([d['X'], d['XE'], d['XS']])
        D[l] = d
    n_g = sum(len(d['gulps']) for d in D.values())
    print(f'{hours:.1f} h, {n_g} gulps, {sum(len(d["A"]) for d in D.values()) / hours:.0f} anchors/h; '
          f'features: A {D[lectures[0]]["XA"].shape[1]}, B+C+D {D[lectures[0]]["XS"].shape[1]}\n')
    print(f'{"model":<28}' + ''.join(f'{"@ " + str(t) + " FA/h":>12}' for t in oc.FA_TARGETS))
    runs = {}
    for name, key, model in [('A logistic (current)', 'XA', 'logistic'),
                             ('A+B+C+D logistic', 'XAS', 'logistic'),
                             ('B+C+D logistic', 'XS', 'logistic'),
                             ('B+C+D boosted', 'XS', 'boosted'),
                             ('A+B+C+D boosted', 'XAS', 'boosted'),
                             ('E logistic (peak patch)', 'XE', 'logistic'),
                             ('A+E logistic', 'XAE', 'logistic'),
                             ('A+E+B+C+D logistic', 'XAES', 'logistic')]:
        runs[name] = cv_scores(D, lectures, key, model)
        row = ra.evaluate(D, runs[name], lectures, hours, oc.FA_TARGETS)
        print(f'{name:<28}' + ''.join(f'{a:>12.0%}' for a, _, _ in row), flush=True)
    for a, b in [('A logistic (current)', 'B+C+D boosted'), ('A logistic (current)', 'A+B+C+D boosted')]:
        mix = {l: (runs[a][l] + runs[b][l]) / 2 for l in lectures}
        row = ra.evaluate(D, mix, lectures, hours, oc.FA_TARGETS)
        print(f'{"avg: " + a.split(" (")[0] + " + " + b:<28}'[:28] + ''.join(f'{v:>12.0%}' for v, _, _ in row))
    print(f'\n(caught share of {n_g} gulps; one gulp = {100 / n_g:.1f} points; 95% CI at 70% about '
          f'+-{196 * np.sqrt(.7 * .3 / n_g):.0f} points)')


if __name__ == '__main__':
    main()
