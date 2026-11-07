"""Result of the blind false-alarm audit (build_audit.py): how many alarms are really gulps,
checked against the controls, and what that means for the measured catch rate.

Usage: .venv\\Scripts\\python tools\\audit_result.py
"""
import json
import os
from collections import Counter

from scipy.stats import beta

import gulpdata as g


def ci(x, n):
    return (beta.ppf(0.025, x, n - x + 1) if x else 0.0, beta.ppf(0.975, x + 1, n - x) if x < n else 1.0)


def main():
    k = json.load(open(os.path.join(g.REVIEW, 'audit_key.json'), encoding='utf-8'))
    d = g.load_decisions()
    by = {}
    for cid, kind in k['key'].items():
        by.setdefault(kind, []).append(d.get(cid, {}).get('label', '(none)'))
    for kind, labs in by.items():
        print(f'{kind:<13}{len(labs):>3}  ' + ', '.join(f'{l} {n}' for l, n in Counter(labs).most_common()))
    al = by['alarm']
    judged = [x for x in al if x not in ('(none)', 'unsure')]
    x, n = sum(l in g.GULP_LABELS for l in judged), len(judged)
    lo, hi = ci(x, n)
    print(f'\nalarms judged gulp: {x}/{n} = {x / n:.0%} (95% CI {lo:.0%}-{hi:.0%}); unsure {al.count("unsure")}, '
          f'not labelled {al.count("(none)")}')
    gc = by['gulp_control']; nc = by['neg_control']
    print(f'controls: known gulps called gulp {sum(l in g.GULP_LABELS for l in gc)}/{len(gc)}, '
          f'low-score sounds called gulp {sum(l in g.GULP_LABELS for l in nc)}/{len(nc)}')
    # What it means at 5 FA/h: of the n_alarms sounds above the threshold, the share p are gulps.
    N, hours = k['n_alarms'], k['hours']
    p = x / n
    print(f'\nat the 5 FA/h threshold: {N} alarms in {hours:.1f} h; about {p * N * k["n_unlabelled"] / N:.0f} of them '
          f'gulps -> true false alarms about {(N - p * k["n_unlabelled"]) / hours:.1f}/h')


if __name__ == '__main__':
    main()
