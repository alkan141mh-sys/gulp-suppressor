"""Read-ahead mute variants, scored against the reviewed gulp edges.

Read-ahead lets a mute start BEFORE the decision that causes it. Variants:
  current       mute from the firing anchor (what live mode can do)
  from onset    a firing secondary anchor mutes from its sound's onset
  + pre-roll    and the mute starts x ms earlier still

Same detections in every variant (only mute starts move), so differences are edge fit only.
Metrics (per hour of the 36 training lectures):
  gulps >=80% duration / energy muted, mute start relative to the gulp's start,
  audible non-gulp audio muted (s/h) and voiced speech muted (s/h).
The extension's model was trained on these lectures: catch rates are optimistic; edge fit is not.

Usage: .venv\\Scripts\\python tools\\eval_mute_variants.py
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import features
import gulpdata
import onset_classifier as oc
from eval_mutes import run_node

HOP = 0.01
VARIANTS = [('current', []), ('from onset', ['--from-onset'])] + \
           [(f'from onset + {p} ms', ['--from-onset', '--preroll', str(p)]) for p in (10, 20, 30, 50)]


def main():
    lectures = [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]
    pre = {}
    for l in lectures:
        f = features.lecture_features(l)
        _, floor = oc.detect_onsets(f['total'])
        tot = f['total']
        T = len(tot)
        gulps = [(s, e) for s, e, lab in gulpdata.load_labels(l) if lab in gulpdata.GULP_LABELS]
        in_g = np.zeros(T, bool)
        for s, e in gulps:
            in_g[int(s / HOP):int(np.ceil(e / HOP))] = True
        fl10 = np.percentile(tot, 10)
        pre[l] = dict(tot=tot, fl10=fl10, gulps=gulps, in_g=in_g, T=T,
                      active=(tot >= fl10 + 10) & ~in_g,
                      voiced=(f['harm'] >= 0.8) & (tot >= floor + 25) & ~in_g)
    hours = sum(p['T'] for p in pre.values()) * HOP / 3600
    print(f'{len(lectures)} lectures, {hours:.1f} h, {sum(len(p["gulps"]) for p in pre.values())} gulps\n')
    print(f'{"variant":<22}{">=80% dur":>10}{">=80% en":>10}{"start-gulp med/p10/p90 ms":>28}{"audible cut s/h":>17}{"voiced cut s/h":>16}')
    for name, extra in VARIANTS:
        with ThreadPoolExecutor(max_workers=6) as ex:
            runs = list(ex.map(lambda l: run_node(gulpdata.wav_path(l), ['--rate', '1'] + extra), lectures))
        dur, en, st, aud, voi = [], [], [], 0.0, 0.0
        for l, r in zip(lectures, runs):
            p = pre[l]
            m = np.zeros(p['T'], bool)
            for a, b in r['mutes']:
                m[max(0, int(a / HOP)):min(p['T'], int(np.ceil(b / HOP)))] = True
            for s, e in p['gulps']:
                k0, k1 = int(s / HOP), max(int(s / HOP) + 1, int(np.ceil(e / HOP)))
                w = np.maximum(10 ** ((p['tot'][k0:k1] - p['fl10']) / 10) - 1, 0)
                dur.append(m[k0:k1].mean()); en.append((w * m[k0:k1]).sum() / max(w.sum(), 1e-9))
                hit = [a for a, b in r['mutes'] if a < e and s < b]
                if hit:
                    st.append((min(hit) - s) * 1000)
            aud += (m & p['active']).sum() * HOP
            voi += (m & p['voiced']).sum() * HOP
        dur, en, st = np.array(dur), np.array(en), np.array(st)
        print(f'{name:<22}{np.mean(dur >= .8):>10.0%}{np.mean(en >= .8):>10.0%}'
              f'{f"{np.median(st):.0f} / {np.percentile(st, 10):.0f} / {np.percentile(st, 90):.0f}":>28}'
              f'{aud / hours:>17.2f}{voi / hours:>16.2f}', flush=True)
    print('\n(start-gulp: when the mute starts relative to the gulp\'s labelled start; negative = before it)')


if __name__ == '__main__':
    main()
