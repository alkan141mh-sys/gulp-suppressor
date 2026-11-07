"""Onset classifier with a second chance at the main burst.

miss_analysis.py showed that missed gulps start with a weak lead-in: the onset detector fires
on it, and the main burst arrives after the classifier's 100 ms window. Here, inside each sound
the detector also places SECONDARY anchors wherever the level climbs JUMP_DB above its highest
value since the onset. Each anchor gets its own patch and decision, still N = 100 ms after the
anchor, so the lookahead stays at 100 ms.

Evaluation unit is the SOUND (an onset plus its secondary anchors):
  gulp caught       some anchor inside the gulp scores above the threshold
  false alarm       a non-gulp sound with any anchor above the threshold (one sound is muted
                    once, however many of its anchors fire), per hour
  energy muted      share of the gulp's energy from its first firing anchor onward (the lead-in
                    before a late anchor is heard)

The baseline is the same code with primary anchors only, i.e. onset_classifier.py.

Usage: .venv\\Scripts\\python tools\\reanchor.py
"""
import numpy as np

import gulpdata
import onset_classifier as oc
from find_candidates import fit_logreg, logit
from output_roc import gulp_edges, n_future

N_MS, LAM = 100, 100.0
JUMP_DB = 6.0          # a secondary anchor: level this far above the running max since the onset
MIN_GAP = 4            # frames between anchors of one sound (40 ms)
MAX_AFTER = 50         # look for jumps up to 500 ms after the onset
END_QUIET = 10         # the sound ends after 100 ms back near the floor
MAX_SECONDARY = 3
# Missed gulps begin with a WEAK lead-in; speech climbs fast. A secondary anchor is only placed
# while the sound has stayed below floor + LEADIN_DB since its onset (None: no such gate).
LEADIN_DB = None


def find_anchors(total, floor, onsets):
    """Anchor frames, the sound (onset group) each belongs to, and 0 = onset / 1 = secondary."""
    A, grp, kind = [], [], []
    for gi, k0 in enumerate(onsets):
        A.append(k0); grp.append(gi); kind.append(0)
        top, last, quiet, n2 = total[k0], k0, 0, 0
        for k in range(k0 + 1, min(len(total), k0 + MAX_AFTER + 1)):
            if total[k] < floor[k0] + oc.ONSET_DB:
                quiet += 1
                if quiet >= END_QUIET:
                    break
            else:
                quiet = 0
            weak = LEADIN_DB is None or top < floor[k0] + LEADIN_DB
            if weak and total[k] >= top + JUMP_DB and k - last >= MIN_GAP and n2 < MAX_SECONDARY:
                A.append(k); grp.append(gi); kind.append(1)
                last, n2 = k, n2 + 1
            top = max(top, total[k])
    return np.array(A), np.array(grp), np.array(kind)


def lecture_data(f, labels):
    total = f['total']
    on, floor = oc.detect_onsets(total)
    A, grp, kind = find_anchors(total, floor, on)
    t = A * oc.HOP_S
    y = np.zeros(len(A), int)
    w = np.ones(len(A))
    gulps = []
    for s, e, lab in labels:
        inside = (t >= s - 0.05) & (t <= e)
        if lab in gulpdata.GULP_LABELS:
            k0, k1, _ = gulp_edges(total, s, e)
            y[inside] = -1
            hit = (A >= k0 - 5) & (A <= k1)
            # Positives: the gulp's first onset (as in onset_classifier.py) and its secondary
            # anchors. Later onsets inside the gulp (after a quiet gap) stay ignored: they look
            # unlike a gulp start, and treating them as positives cost 15 points at 5 FA/h.
            prim = np.flatnonzero(hit & (kind == 0))
            if len(prim):
                y[prim[0]] = 1
            y[hit & (kind == 1)] = 1
            gulps.append({'label': lab, 'k0': k0, 'k1': k1, 'anchors': np.flatnonzero(hit)})
        elif lab in gulpdata.IGNORE_LABELS:
            y[inside] = -1
        elif lab in gulpdata.KEEP_LABELS:
            w[inside & (y == 0)] = oc.HARD_NEG_WEIGHT
    # Patch at each anchor (the floor tracker still holds the room level there: it is a minimum
    # over the second before), plus whether it is a secondary anchor and how late it comes.
    X = oc.patches(f, A, floor, n_future(N_MS))
    since = (A - on[grp]) * oc.HOP_S * 1000
    X = np.hstack([X, kind[:, None].astype(float), since[:, None]])
    return {'X': X, 'y': y, 'w': w, 'grp': grp, 'kind': kind, 'A': A, 'gulps': gulps}


def merged_gulps(d, gap_frames=10):
    """Gulps for counting: labels less than 100 ms apart are one gulp (a gulp drawn as two
    parts in review). Returns lists of anchor indices, one per merged gulp."""
    gs = sorted(d['gulps'], key=lambda g: g['k0'])
    out = []
    for g in gs:
        if out and g['k0'] - out[-1]['k1'] <= gap_frames:
            out[-1]['k1'] = max(out[-1]['k1'], g['k1'])
            out[-1]['anchors'] = np.concatenate([out[-1]['anchors'], g['anchors']])
        else:
            out.append({'k0': g['k0'], 'k1': g['k1'], 'anchors': np.asarray(g['anchors']), 'label': g['label']})
    return out


def evaluate(D, scores, lectures, hours, targets):
    """P_D, energy muted and FA/h per sound, for each FA target."""
    gs, neg = [], []
    for l in lectures:
        d, z = D[l], scores[l]
        for g in d['gulps']:
            a = g['anchors']
            gs.append((z[a].max() if len(a) else -np.inf, z[a] if len(a) else np.array([]), d['A'][a], g, l))
        ok = d['y'] == 0
        bad = np.zeros(d['grp'].max() + 1 if len(d['grp']) else 0, bool)
        bad[d['grp'][d['y'] != 0]] = True                          # sounds touching a gulp or ignore span
        best = np.full(len(bad), -np.inf)
        np.maximum.at(best, d['grp'][ok], z[ok])
        neg.append(best[~bad & np.isfinite(best)])
    neg = np.sort(np.concatenate(neg))
    out = []
    for t in targets:
        k = int(t * hours)
        eta = neg[-k - 1] if k < len(neg) else -np.inf
        caught, muted = [], []
        for top, za, aa, g, l in gs:
            c = top > eta
            caught.append(c)
            if c:
                first = aa[np.argmax(za > eta)]
                tot = D[l]['total']
                seg = tot[g['k0']:g['k1'] + 1]
                fl = np.percentile(tot, 10)
                en = np.maximum(10 ** ((seg - fl) / 10) - 1, 0)
                muted.append(en[max(0, first - g['k0']):].sum() / max(en.sum(), 1e-9))
            else:
                muted.append(0.0)
        out.append((np.mean(caught), np.median([m for m, c in zip(muted, caught) if c]) if any(caught) else 0.0,
                    np.mean(np.array(muted) >= 0.8)))
    return out


def cv(D, lectures, primary_only):
    scores = {}
    for fold in [lectures[i::oc.N_FOLDS] for i in range(oc.N_FOLDS)]:
        tr = [l for l in lectures if l not in fold]
        sel = {l: (D[l]['y'] >= 0) & ((D[l]['kind'] == 0) | (not primary_only)) for l in tr}
        Xt = np.vstack([D[l]['X'][sel[l]] for l in tr])
        yt = np.concatenate([D[l]['y'][sel[l]] for l in tr]).astype(float)
        wt = np.concatenate([D[l]['w'][sel[l]] for l in tr])
        wt[yt == 1] *= wt[yt == 0].sum() / wt[yt == 1].sum()
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        beta = fit_logreg(np.clip((Xt - mu) / sd, -4, 4), yt, wt, lam=LAM)
        for l in fold:
            z = logit(beta, np.clip((D[l]['X'] - mu) / sd, -4, 4))
            if primary_only:
                z = np.where(D[l]['kind'] == 0, z, -np.inf)
            scores[l] = z
    return scores


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {}
    for l in lectures:
        D[l] = lecture_data(feats[l], gulpdata.load_labels(l))
        D[l]['total'] = feats[l]['total']
    n_a = sum(len(d['A']) for d in D.values())
    n_s = sum(int((d['kind'] == 1).sum()) for d in D.values())
    n_g = sum(len(d['gulps']) for d in D.values())
    with_sec = sum(any(D[l]['kind'][g['anchors']] == 1) for l in lectures for g in D[l]['gulps'])
    print(f'{hours:.1f} h, {n_g} gulps; anchors {n_a / hours:.0f}/h of which secondary {n_s / hours:.0f}/h; '
          f'gulps with a secondary anchor inside: {with_sec}\n')
    print(f'{"detector":<24}' + ''.join(f'{"@ " + str(t) + " FA/h":>30}' for t in oc.FA_TARGETS))
    print(f'{"":<24}' + '   caught  energy(caught)  >=80%' * len(oc.FA_TARGETS))
    for name, prim in [('first onset only', True), ('with re-anchoring', False)]:
        row = evaluate(D, cv(D, lectures, prim), lectures, hours, oc.FA_TARGETS)
        print(f'{name:<24}' + ''.join(f'{a:>9.0%}{b:>15.0%}{c:>7.0%}' for a, b, c in row), flush=True)
    print('  (caught = some anchor in the gulp fired; energy(caught) = median share of a caught gulp\'s '
          'energy muted; >=80% = share of ALL gulps with at least 80% of their energy muted)')


if __name__ == '__main__':
    main()
