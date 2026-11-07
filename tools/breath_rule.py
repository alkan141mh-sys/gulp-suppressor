"""The user's breath-vs-gulp test, for sounds that could be either.

Gate (only these sounds are tested; everything else is untouched): no voiced frame and peak
below 35 dB above the floor in the first 400 ms of the sound.
Measured over the sound (onset until 100 ms back near the floor, at most 400 ms):
  tilt      dB of the energy above 3.2 kHz minus dB of the energy below it (breath: hissy, high)
  lf_peak   peak level of the energy below 3.2 kHz, dB above the floor (gulp: a low thump)
  lf_rough  mean frame-to-frame change of that low-band level, dB per 10 ms (gulp: bursty;
            breath: steady)
  lf_cstd   standard deviation of the low-band spectral centroid over the sound, Hz
The user's rule: a sound is a breath when it is hissy AND weak below 3.2 kHz, OR when its low
band is steady; a sound that is low-frequency only (not hissy) is judged by the last two.

Labelled gulps vs every reviewed non-gulp label (breath, none, not_gulp: from all the review
queues), lecture by lecture. Limits are chosen to lose at most 1% of the gulps in the gate.

Usage: .venv\\Scripts\\python tools\\breath_rule.py
"""
import numpy as np

import features
import gulpdata as g
import onset_classifier as oc

LF_BANDS = 19            # bands 0..18 end at 3203 Hz
MAX_FR = 40


def sound_feats(f, floor, k0):
    tot, harm, B = f['total'], f['harm'], f['bands']
    T = len(tot)
    fl = floor[k0]
    quiet, end = 0, min(T, k0 + MAX_FR)
    for k in range(k0 + 1, min(T, k0 + MAX_FR)):
        if tot[k] < fl + oc.ONSET_DB:
            quiet += 1
            if quiet >= 10:
                end = k - 9
                break
        else:
            quiet = 0
    end = max(end, k0 + 2)
    P = 10 ** (B[k0:end].astype(np.float64) / 10)
    lf = 10 * np.log10(P[:, :LF_BANDS].sum(1) + 1e-12)
    hf = 10 * np.log10(P[:, LF_BANDS:].sum(1) + 1e-12)
    w40 = slice(k0, min(T, k0 + MAX_FR))
    lev40 = tot[w40] - fl
    voiced = int(((harm[w40] >= 0.8) & (lev40 >= 25)).sum())
    cf = (features.BAND_CENTRES[:LF_BANDS] if hasattr(features, 'BAND_CENTRES') else
          (np.array(features.BAND_EDGES[:LF_BANDS]) + np.array(features.BAND_EDGES[1:LF_BANDS + 1])) / 2 * features.FS / features.NFFT)
    cent = (P[:, :LF_BANDS] * cf).sum(1) / P[:, :LF_BANDS].sum(1)
    act = lf >= fl + 3
    return {'gate': voiced == 0 and lev40.max() < 35,
            'tilt': 10 * np.log10(P[:, LF_BANDS:].sum() + 1e-12) - 10 * np.log10(P[:, :LF_BANDS].sum() + 1e-12),
            'lf_peak': lf.max() - fl,
            'lf_rough': np.abs(np.diff(lf)).mean() if len(lf) > 1 else 0.0,
            'lf_cstd': cent[act].std() if act.sum() > 1 else 0.0,
            'dur': (end - k0) * 10}


def collect():
    G, N = [], []
    for l in g.lectures():
        if l in g.TEST_LECTURES:
            continue
        f = features.lecture_features(l)
        on, floor = oc.detect_onsets(f['total'])
        t = on * oc.HOP_S
        for s, e, lab in g.load_labels(l):
            hit = np.flatnonzero((t >= s - 0.05) & (t <= e))
            if not len(hit):
                continue
            x = sound_feats(f, floor, int(on[hit[0]]))
            x['lab'], x['where'] = lab, f'{l} {s:.2f}'
            if lab in g.GULP_LABELS:
                G.append(x)
            elif lab in ('breath', 'none', 'not_gulp', 'mouth'):
                N.append(x)
    return G, N


def main():
    G, N = collect()
    Gg = [x for x in G if x['gate']]; Ng = [x for x in N if x['gate']]
    print(f'{len(G)} gulps ({len(Gg)} in the gate), {len(N)} reviewed non-gulps ({len(Ng)} in the gate: '
          f'{sum(x["lab"] == "breath" for x in Ng)} breath, {sum(x["lab"] != "breath" for x in Ng)} other)\n')
    for k in ('tilt', 'lf_peak', 'lf_rough', 'lf_cstd', 'dur'):
        a = np.array([x[k] for x in Gg]); b = np.array([x[k] for x in Ng if x['lab'] == 'breath'])
        c = np.array([x[k] for x in Ng if x['lab'] != 'breath'])
        q = lambda v: f'{np.percentile(v, 1):.1f} / {np.percentile(v, 5):.1f} / {np.median(v):.1f} / {np.percentile(v, 95):.1f} / {np.percentile(v, 99):.1f}'
        # AUC: P(gulp value > non-gulp value)
        auc = lambda p, n: (np.mean(p[:, None] > n[None, :]) + 0.5 * np.mean(p[:, None] == n[None, :])) if len(n) else np.nan
        print(f'{k:<9} gulps p1/p5/p50/p95/p99 {q(a)}\n{"":<9} breath             {q(b) if len(b) else "-"}\n'
              f'{"":<9} other              {q(c) if len(c) else "-"}\n{"":<9} AUC gulp>breath {auc(a, b):.2f}, gulp>other {auc(a, c):.2f}\n')
    return G, N


if __name__ == '__main__':
    main()
