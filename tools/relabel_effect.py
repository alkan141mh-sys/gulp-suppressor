"""Effect of the confirmed-gulp review pass (redrawn edges, not_gulp, two_sounds) on the detector.

Runs the same lecture-grouped cross-validation as reanchor.py twice:
  before   labels without the confirmed-queue notes (conf_* decisions excluded)
  after    labels with them
and prints how the labels changed plus caught / energy-muted at each FA/h target.

Usage: .venv\\Scripts\\python tools\\relabel_effect.py
"""
import numpy as np

import gulpdata
import onset_classifier as oc
import reanchor as ra
from output_roc import gulp_edges

BEFORE = lambda cid: cid.startswith('conf_')


def label_change(lectures):
    dur_b, dur_a, n_b, n_a, ng = [], [], 0, 0, 0
    for l in lectures:
        b = [x for x in gulpdata.load_labels(l, BEFORE) if x[2] in gulpdata.GULP_LABELS]
        a = [x for x in gulpdata.load_labels(l) if x[2] in gulpdata.GULP_LABELS]
        n_b += len(b); n_a += len(a)
        dur_b += [e - s for s, e, _ in b]; dur_a += [e - s for s, e, _ in a]
        ng += sum(1 for x in gulpdata.load_labels(l) if x[2] == 'not_gulp')
    d = gulpdata.load_decisions()
    tags = [t for k, v in d.items() if k.startswith('conf_') for t in v.get('tags', [])]
    print(f'confirmed-pass notes: ' + ', '.join(f'{t} {tags.count(t)}' for t in sorted(set(tags))))
    print(f'gulp labels: before {n_b}, after {n_a}')
    print(f'label length: median {np.median(dur_b)*1000:.0f} -> {np.median(dur_a)*1000:.0f} ms, '
          f'total {sum(dur_b):.1f} -> {sum(dur_a):.1f} s')


def run(lectures, feats, hours, exclude):
    D = {}
    for l in lectures:
        D[l] = ra.lecture_data(feats[l], gulpdata.load_labels(l, exclude))
        D[l]['total'] = feats[l]['total']
    ng = sum(len(d['gulps']) for d in D.values())
    rows = {}
    for name, prim in [('first onset only', True), ('with re-anchoring', False)]:
        rows[name] = ra.evaluate(D, ra.cv(D, lectures, prim), lectures, hours, oc.FA_TARGETS)
    return ng, rows


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    label_change(lectures)
    print(f'\n{hours:.1f} h of training lectures, 6-fold lecture-grouped CV\n')
    print(f'{"":<34}' + ''.join(f'{"@ " + str(t) + " FA/h":>22}' for t in oc.FA_TARGETS))
    print(f'{"":<34}' + '   caught    >=80% en.' * len(oc.FA_TARGETS))
    for tag, ex in [('before', BEFORE), ('after', None)]:
        ng, rows = run(lectures, feats, hours, ex)
        for name, row in rows.items():
            print(f'{tag + f" ({ng} gulps) " + name:<34}' + ''.join(f'{a:>9.0%}{c:>13.0%}' for a, b, c in row),
                  flush=True)


if __name__ == '__main__':
    main()
