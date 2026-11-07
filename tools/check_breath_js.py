"""Testbench: engine.js's breath rule against breath_rule.py, decision by decision.

Runs the JS engine (run_lecture.mjs --breath) over a lecture and compares, for every decided
anchor, whether JS vetoed it as a breath with what the Python rule says about its sound.

Usage: .venv\\Scripts\\python tools\\check_breath_js.py [lecture]
"""
import sys

import numpy as np

import features
import gulpdata as g
import onset_classifier as oc
from train_final import shape_veto
from eval_mutes import run_node


def main():
    lec = sys.argv[1] if len(sys.argv) > 1 else '005'
    r = run_node(g.wav_path(lec), ['--fa', '20', '--from-onset', '--preroll', '30', '--veto-frames', '20',
                                   '--veto-db', '43', '--breath'])
    f = features.lecture_features(lec)
    on, floor = oc.detect_onsets(f['total'])
    py = {}
    for k0 in on:
        py[int(k0)] = shape_veto(f, floor, int(k0))
    agree = n = js_b = py_b = 0
    bad = []
    for t, kind, z, fired, veto in r['events']:
        if kind != 0 or veto == 1:
            continue                      # primary anchors only (their k0 is the frame itself); skip level vetoes
        k = int(round(t / 0.01))
        if k not in py:
            continue
        n += 1
        jb, pb = veto == 2, py[k]
        js_b += jb; py_b += pb
        if jb == pb:
            agree += 1
        elif len(bad) < 5:
            bad.append((t, jb, pb))
    print(f'lecture {lec}: {n} onsets compared, breath veto JS {js_b} / Python {py_b}, agree {agree}/{n}')
    for b in bad:
        print('  disagree at', b)
    print('PASS' if agree == n else 'FAIL')


if __name__ == '__main__':
    main()
