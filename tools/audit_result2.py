"""Round-2 audit result: gulp share per score band, controls, and the corrected false-alarm rate
at the measured 5 and 20 FA/h thresholds (bands from build_audit2.py).

Usage: .venv\\Scripts\\python tools\\audit_result2.py
"""
import json
import os
from collections import Counter

from scipy.stats import beta

import gulpdata as g


def ci(x, n):
    return (beta.ppf(0.025, x, n - x + 1) if x else 0.0, beta.ppf(0.975, x + 1, n - x) if x < n else 1.0)


def main():
    K = json.load(open(os.path.join(g.REVIEW, 'audit_key.json'), encoding='utf-8'))
    r1 = set(json.load(open(os.path.join(g.REVIEW, 'audit_key.round1.json'), encoding='utf-8'))['key'])
    d = g.load_decisions()
    lab = lambda c: d.get(c, {}).get('label', '(none)')
    r2 = {c: k for c, k in K['key'].items() if c not in r1}
    for kind in ('gulp_control', 'neg_control'):
        labs = [lab(c) for c, k in r2.items() if k == kind]
        print(f'{kind:<13} ' + ', '.join(f'{l} {n}' for l, n in Counter(labs).most_common()))
    R = K['round2']
    share = {}
    for band in ('A', 'B'):
        labs = [lab(c) for c, b in K['band'].items() if b == band]
        judged = [x for x in labs if x not in ('unsure', '(none)')]
        x, n = sum(l in g.GULP_LABELS for l in judged), len(judged)
        lo, hi = ci(x, n)
        share[band] = x / n
        print(f'band {band}: {x}/{n} gulp = {x / n:.0%} (95% CI {lo:.0%}-{hi:.0%}); ' +
              ', '.join(f'{l} {c}' for l, c in Counter(labs).most_common()))
    h = R['hours']
    # labelled (non-gulp) sounds in a band are true false alarms; unlabelled ones are gulps with the band's share
    fa5 = (R['nA'] - R['unlabelled_A'] * share['A']) / h
    fa20 = fa5 + (R['nB'] - R['unlabelled_B'] * share['B']) / h
    hid = R['unlabelled_A'] * share['A'] + R['unlabelled_B'] * share['B']
    print(f'\ncorrected false alarms: measured-5 FA/h threshold -> about {fa5:.1f}/h; measured-20 -> about {fa20:.1f}/h')
    print(f'estimated unlabelled gulps above the 20 FA/h threshold: about {hid:.0f} (in {h:.1f} h)')


if __name__ == '__main__':
    main()
