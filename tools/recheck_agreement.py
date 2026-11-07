"""How consistent is the labelling? Compare blind re-check labels with the originals.

Cohen's kappa corrects raw agreement for chance:  kappa = (p_o - p_e) / (1 - p_e),
where p_o is observed agreement and p_e the agreement expected if both passes picked
labels independently with their own label frequencies. The weighted kappa treats the
severity scale H0 < gulp_low < gulp < gulp_high as ordinal, so low-vs-gulp counts as a
smaller disagreement than low-vs-high (linear weights |i - j| / (k - 1)).

Usage: .venv\\Scripts\\python tools\\recheck_agreement.py [--from 100 --to 160]
"""
import argparse
import json
import os

import numpy as np

import gulpdata

ORDINAL = ['H0', 'gulp_low', 'gulp', 'gulp_high']


def category(label):
    return 'H0' if label in gulpdata.KEEP_LABELS else label


def kappa(a, b, cats, weighted=False):
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.indices(O.shape)
    W = np.abs(i - j) / (len(cats) - 1) if weighted else (i != j).astype(float)
    return 1 - (W * O).sum() / (W * E).sum(), O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='lo', type=int, default=1, help='first candidate rank to include')
    ap.add_argument('--to', dest='hi', type=int, default=10 ** 9, help='last candidate rank to include')
    args = ap.parse_args()
    with open(os.path.join(gulpdata.REVIEW, 'candidates.json'), encoding='utf-8') as f:
        rank = {c['id']: c['rank'] for c in json.load(f)['candidates']}

    pairs, shifts = [], []
    for cid, d in gulpdata.load_decisions().items():
        if not d.get('rechecked') or not args.lo <= rank.get(cid, 0) <= args.hi:
            continue
        before = [h for h in d.get('history', []) if not h.get('rechecked')]
        if not before:
            continue
        o = before[-1]
        pairs.append((category(o['label']), category(d['label'])))
        shifts.append((d['start'] - o['start'], d['end'] - o['end']))
    if not pairs:
        print('no re-checked decisions yet')
        return
    a, b = zip(*pairs)
    cats = sorted(set(a) | set(b), key=lambda c: (ORDINAL.index(c) if c in ORDINAL else 99, c))
    k, O = kappa(a, b, cats)
    print(f'{len(pairs)} re-checked; raw agreement {np.mean([x == y for x, y in pairs]):.0%}, Cohen kappa {k:.2f}')
    ordinal = [(x, y) for x, y in pairs if x in ORDINAL and y in ORDINAL]
    if len(ordinal) >= 2 and len({x for x, _ in ordinal} | {y for _, y in ordinal}) >= 2:
        kw, _ = kappa(*zip(*ordinal), ORDINAL, weighted=True)
        print(f'weighted kappa on the severity scale ({len(ordinal)} pairs): {kw:.2f}')
    print('\nrows = first pass, columns = blind re-check (counts)')
    n = len(pairs)
    print(' ' * 13 + ''.join(f'{c:>13}' for c in cats))
    for i, c in enumerate(cats):
        print(f'{c:>13}' + ''.join(f'{round(O[i, j] * n):>13}' for j in range(len(cats))))
    s = np.abs(np.array(shifts)) * 1000
    print(f'\nedge changes: start median {np.median(s[:, 0]):.0f} ms, end median {np.median(s[:, 1]):.0f} ms')


if __name__ == '__main__':
    main()
