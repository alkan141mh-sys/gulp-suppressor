"""How well do the extension's mute intervals fit the gulps, measured against the label edges
themselves (no re-measuring from the audio)? Runs extension/engine.js once per lecture and
scores the same mutes against the labels before and after the confirmed-gulp review pass.

Per gulp:     share of its duration and of its energy (above the 10th-percentile level) muted.
Per mute that touches a gulp:
  early       how long the mute starts before the gulp (cuts whatever came before it)
  overrun     how long it lasts after the gulp ends
  overrun into voiced speech: frames after the gulp, inside the mute, that are voiced speech
              (harmonicity >= 0.8, >= floor + 25 dB) -> the 'swallowed the next word' problem
Mutes touching no gulp are false alarms; their length is reported too.

The extension's model was trained on these lectures, so catch rates are optimistic; the edge
fit (early / overrun) does not depend much on that.

Usage: .venv\\Scripts\\python tools\\eval_mute_edges.py
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import features
import gulpdata
import onset_classifier as oc
from eval_mutes import run_node
from relabel_effect import BEFORE

HOP = 0.01


def score(lec, mutes, labels, tot, harm, floor):
    T = len(tot)
    fl10 = np.percentile(tot, 10)
    voiced = (harm >= 0.8) & (tot >= floor + 25)
    gulps = [(s, e) for s, e, lab in labels if lab in gulpdata.GULP_LABELS]
    m = np.zeros(T, bool)
    for a, b in mutes:
        m[max(0, int(a / HOP)):min(T, int(np.ceil(b / HOP)))] = True
    per_gulp = []
    for s, e in gulps:
        k0, k1 = int(s / HOP), max(int(s / HOP) + 1, int(np.ceil(e / HOP)))
        en = np.maximum(10 ** ((tot[k0:k1] - fl10) / 10) - 1, 0)
        per_gulp.append((m[k0:k1].mean(), (en * m[k0:k1]).sum() / max(en.sum(), 1e-9)))
    per_mute, fa = [], []
    in_gulp = np.zeros(T, bool)
    for s, e in gulps:
        in_gulp[int(s / HOP):int(np.ceil(e / HOP))] = True
    for a, b in mutes:
        hit = [(s, e) for s, e in gulps if s < b and a < e]
        if not hit:
            fa.append(b - a)
            continue
        s0, e1 = min(s for s, _ in hit), max(e for _, e in hit)
        ka, kb = int(a / HOP), int(np.ceil(b / HOP))
        over = np.zeros(T, bool)
        over[ka:kb] = True
        over &= ~in_gulp
        over[:int(s0 / HOP)] = False                    # only after the gulp's start
        per_mute.append((s0 - a, b - e1, (over & voiced).sum() * HOP, ((over & voiced).sum() > 0)))
    return per_gulp, per_mute, fa, T * HOP


def main():
    lectures = [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]
    with ThreadPoolExecutor(max_workers=6) as ex:
        runs = list(ex.map(lambda l: run_node(gulpdata.wav_path(l), ['--rate', '1']), lectures))
    for tag, ex_ in [('old edges', BEFORE), ('new edges', None)]:
        G, M, FA, secs = [], [], [], 0.0
        for l, r in zip(lectures, runs):
            f = features.lecture_features(l)
            _, floor = oc.detect_onsets(f['total'])
            g, m, fa, s = score(l, r['mutes'], gulpdata.load_labels(l, ex_), f['total'], f['harm'], floor)
            G += g; M += m; FA += fa; secs += s
        G, M = np.array(G), np.array(M)
        h = secs / 3600
        print(f'\n== {tag}: {len(G)} gulps, {len(M)} mutes on gulps, {len(FA)} mutes elsewhere ({len(FA)/h:.1f}/h)')
        print(f'  gulps muted >=80% of duration {np.mean(G[:, 0] >= .8):.0%}, >=80% of energy {np.mean(G[:, 1] >= .8):.0%}, '
              f'any {np.mean(G[:, 0] > 0):.0%}')
        print(f'  mute starts before the gulp: median {np.median(M[:, 0])*1000:.0f} ms, p90 {np.percentile(M[:, 0], 90)*1000:.0f} ms')
        print(f'  mute ends after the gulp:    median {np.median(M[:, 1])*1000:.0f} ms, p90 {np.percentile(M[:, 1], 90)*1000:.0f} ms, '
              f'share > 100 ms {np.mean(M[:, 1] > .1):.0%}')
        print(f'  mutes on gulps that cut voiced speech after it: {M[:, 3].mean():.0%} '
              f'(total {M[:, 2].sum():.1f} s = {M[:, 2].sum()/h:.2f} s/h)')
        print(f'  false-alarm mutes: median length {np.median(FA)*1000:.0f} ms, total {sum(FA)/h:.1f} s/h')


if __name__ == '__main__':
    main()
