"""Train the detector used by live.py and export everything it needs to models/detector.json.

Model: the re-anchoring onset classifier (reanchor.py, N = 100 ms, logistic, lambda 100),
trained on all 36 training lectures. The test lectures stay untouched.

Thresholds: one per TRUE false-alarm rate, taken from the CROSS-VALIDATED sound scores (each
lecture scored by a model that never saw it), as the read-ahead extension decides:
  - the speech veto (200 ms, 43 dB) and the breath rule (breath_rule.py) are applied first;
  - an unlabelled sound above the threshold counts as a false alarm only with the probability
    the blind audits measured for its depth (corrected_roc.py): 1 - 0.90 in the top
    ~5 measured-FA/h worth of sounds, 1 - 0.50 down to ~20, 1 below (conservative).
Live mode has no speech veto or breath rule, so it gives more false alarms at the same setting.

Usage: .venv\\Scripts\\python tools\\train_final.py
"""
import json
import os

import numpy as np

import gulpdata
import onset_classifier as oc
import reanchor as ra
from breath_rule import sound_feats
from find_candidates import fit_logreg
from output_roc import n_future

FA_RATES = [0.5, 1, 2, 3, 5]          # TRUE false alarms per hour (read-ahead mode)
DEFAULT_FA = '2'
VETO_FRAMES, VETO_DB = 20, 43
DS_FRAMES = 30
P_TOP, P_MID = 0.90, 0.50
OUT = os.path.join(gulpdata.ROOT, 'models', 'detector.json')


def shape_veto(f, floor, k0):
    """engine.js _breath: the breath rule, or a weak click that ends before speech (< 400 ms)."""
    x = sound_feats(f, floor, k0)
    seg = f['total'][k0:k0 + 40] - floor[k0]
    hit = np.flatnonzero(seg >= 43)
    if len(hit) and x['dur'] < hit[0] * 10 and x['lf_peak'] < 14:
        return True
    return x['gate'] and x['lf_rough'] < 2.0 and (x['tilt'] > -18 or x['lf_peak'] < 16)


def vetoed(D, feats, lectures, scores):
    """Scores with the read-ahead vetoes applied (-inf for vetoed anchors)."""
    out = {}
    for l in lectures:
        tot = D[l]['total']
        on, floor = oc.detect_onsets(tot)
        A = D[l]['A']
        idx = np.clip(A[:, None] + np.arange(VETO_FRAMES)[None, :], 0, len(tot) - 1)
        speech = tot[idx].max(1) - floor[A] > VETO_DB
        # End of the recording: digital silence within 300 ms after the anchor (no gulp has any
        # within 1 s); the detector otherwise takes the cut to silence for a sound.
        di = np.clip(A[:, None] + np.arange(DS_FRAMES)[None, :], 0, len(tot) - 1)
        speech |= (tot[di] <= oc.DIGITAL_SILENCE_DB).any(1)
        breath = np.array([shape_veto(feats[l], floor, int(k0)) for k0 in on], bool)
        out[l] = np.where(speech | breath[D[l]['grp']], -np.inf, scores[l])
    return out


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {l: {**ra.lecture_data(feats[l], gulpdata.load_labels(l)), 'total': feats[l]['total']} for l in lectures}

    # Thresholds from cross-validated scores, after the vetoes, with audit-corrected false alarms.
    cvs = vetoed(D, feats, lectures, ra.cv(D, lectures, primary_only=False))
    neg, pos = [], []
    for l in lectures:
        d, z = D[l], cvs[l]
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
        for g in ra.merged_gulps(d):                    # a gulp drawn as two parts counts once
            a = g['anchors']
            pos.append(z[a].max() if len(a) else -np.inf)
    neg.sort(key=lambda x: -x[0])
    pos = np.array(pos)
    n5, n20 = int(5 * hours), int(20 * hours)
    w = np.array([1.0 if lab else (1 - P_TOP if r < n5 else 1 - P_MID if r < n20 else 1.0)
                  for r, (_, lab) in enumerate(neg)])
    s = np.array([x[0] for x in neg])
    cum = np.cumsum(w) / hours
    thresholds, caught = {}, {}
    for t in FA_RATES:
        r = int(np.searchsorted(cum, t, side='right'))
        eta = float(s[r]) if r < len(s) else -1e9
        thresholds[str(t)], caught[str(t)] = eta, float(np.mean(pos > eta))

    # Final model on everything.
    sel = {l: D[l]['y'] >= 0 for l in lectures}
    X = np.vstack([D[l]['X'][sel[l]] for l in lectures])
    y = np.concatenate([D[l]['y'][sel[l]] for l in lectures]).astype(float)
    w = np.concatenate([D[l]['w'][sel[l]] for l in lectures])
    w[y == 1] *= w[y == 0].sum() / w[y == 1].sum()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    beta = fit_logreg(np.clip((X - mu) / sd, -4, 4), y, w, lam=ra.LAM)

    model = {
        'description': 'Re-anchoring onset classifier, N = 100 ms, trained on lectures 001-036 '
                       '(reviewed labels incl. both blind audits)',
        'params': {'N_MS': ra.N_MS, 'n_after': n_future(ra.N_MS), 'PRE': oc.PRE, 'ONSET_DB': oc.ONSET_DB,
                   'QUIET_MS': oc.QUIET_MS, 'FLOOR_N': oc.FLOOR_N, 'DIGITAL_SILENCE_DB': oc.DIGITAL_SILENCE_DB,
                   'JUMP_DB': ra.JUMP_DB, 'MIN_GAP': ra.MIN_GAP, 'MAX_AFTER': ra.MAX_AFTER,
                   'END_QUIET': ra.END_QUIET, 'MAX_SECONDARY': ra.MAX_SECONDARY, 'LEADIN_DB': ra.LEADIN_DB},
        # Speech-level reference (95th percentile of frame level) until the live tracker has 60 s.
        'p95_default': float(np.median([np.percentile(feats[l]['total'], 95) for l in lectures])),
        'mu': mu.tolist(), 'sd': sd.tolist(), 'beta': beta.tolist(),
        'thresholds_by_fa_per_h': thresholds, 'default_fa': DEFAULT_FA,
        'fa_meaning': 'true false alarms per hour in read-ahead mode (audit-corrected, after the vetoes)',
        'cv_caught_by_fa_per_h': caught,
        'n_gulps': int(sum(len(d['gulps']) for d in D.values())), 'n_gulps_counted': int(len(pos)), 'hours': hours,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(model, f, indent=1)
    print(f'saved {os.path.relpath(OUT, gulpdata.ROOT)}: {len(beta) - 1} features, {model["n_gulps"]} gulp labels '
          f'= {model["n_gulps_counted"]} gulps (parts < 100 ms apart counted once)')
    for t in FA_RATES:
        print(f'  {t:>3} true FA/h: threshold z = {thresholds[str(t)]:6.2f}, cross-validated caught {caught[str(t)]:.0%}')


if __name__ == '__main__':
    main()
