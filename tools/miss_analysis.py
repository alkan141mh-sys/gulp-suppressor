"""Why are gulps missed? Measure each gulp from the audio and compare caught with missed ones.

The detector is the onset classifier (N = 100 ms, lambda 100) with cross-validated scores; a
gulp counts as caught if its onset scores above the threshold that gives 5 false alarms/h.

Per gulp, from the audio:
  rise_db         peak level above the room floor
  speech_before   loudest level 100-400 ms before the onset, above the floor (a gulp right
                  after speech: the onset detector needs 100 ms of quiet before an onset)
  first100_share  share of the gulp's energy inside its first 100 ms (the classifier's view)
  peak_ms         time from onset to the loudest frame
  bursts          separate core bursts (gaps > 150 ms)
  harm100         mean harmonicity over the first 100 ms (voice-like if high)
  duration_ms     measured gulp length

Usage: .venv\\Scripts\\python tools\\miss_analysis.py
"""
import numpy as np

import gulpdata
import onset_classifier as oc
from output_roc import gulp_edges

N_MS, LAM, FA_PER_H = 100, 100.0, 5
HOP_MS = oc.HOP_S * 1000


def describe(total, harm, floor_at, k0, k1, core):
    seg = total[k0:k1 + 1]
    fl = floor_at
    en = np.maximum(10 ** ((seg - fl) / 10) - 1, 0)
    n100 = int(100 / HOP_MS)
    before = total[max(0, k0 - 40):max(0, k0 - 10)]
    gaps = np.diff(core) * HOP_MS if len(core) > 1 else np.array([])
    return {
        'rise_db': float(seg.max() - fl),
        'speech_before': float(before.max() - fl) if len(before) else 0.0,
        'first100_share': float(en[:n100].sum() / max(en.sum(), 1e-9)),
        'peak_ms': float(np.argmax(seg) * HOP_MS),
        'bursts': int(1 + (gaps > 150).sum()),
        'harm100': float(harm[k0:k0 + n100].mean()),
        'duration_ms': float((k1 - k0 + 1) * HOP_MS),
    }


def main():
    lectures, feats, base = oc.load_all()
    scores = oc.cv_scores(N_MS, LAM, lectures, feats, base)
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    neg = np.sort(np.concatenate([scores[l][base[l][2] == 0] for l in lectures]))
    eta = neg[-int(FA_PER_H * hours) - 1]

    rows = []
    for l in lectures:
        on, floor, y, w, gulps = base[l]
        total, harm = feats[l]['total'], feats[l]['harm']
        glabs = [x for x in gulpdata.load_labels(l) if x[2] in gulpdata.GULP_LABELS]
        for g, (s, e, lab) in zip(gulps, glabs):
            k0, k1, core = gulp_edges(total, s, e)
            i = g['onset_index']
            r = describe(total, harm, floor[k0], k0, k1, core)
            r.update(lec=l, t=s, label=lab, stage1=i is not None,
                     score=float(scores[l][i]) if i is not None else -np.inf)
            r['caught'] = r['stage1'] and r['score'] > eta
            rows.append(r)

    n = len(rows)
    miss = [r for r in rows if not r['caught']]
    print(f'{n} gulps; caught at {FA_PER_H} FA/h: {n - len(miss)} ({(n - len(miss)) / n:.0%}); missed {len(miss)}')
    print(f'  never found by the onset detector: {sum(not r["stage1"] for r in rows)}\n')

    print('miss rate by condition (each row: gulps with that property, and how many of them are missed)')
    conds = [
        ('speech within 100-400 ms before (>= 20 dB over floor)', lambda r: r['speech_before'] >= 20),
        ('quiet: rise < 20 dB over floor', lambda r: r['rise_db'] < 20),
        ('loud: rise >= 30 dB', lambda r: r['rise_db'] >= 30),
        ('late peak: < 40% of energy in first 100 ms', lambda r: r['first100_share'] < 0.4),
        ('more than one burst', lambda r: r['bursts'] > 1),
        ('voice-like first 100 ms (harmonicity >= 0.7)', lambda r: r['harm100'] >= 0.7),
        ('long: > 400 ms', lambda r: r['duration_ms'] > 400),
    ] + [(f'label {lab}', (lambda lab: lambda r: r['label'] == lab)(lab)) for lab in ['gulp_low', 'gulp', 'gulp_high']]
    overall = len(miss) / n
    for name, f in conds:
        sub = [r for r in rows if f(r)]
        if not sub:
            continue
        m = sum(not r['caught'] for r in sub)
        print(f'  {name:<56} {len(sub):>4} gulps, missed {m:>3} ({m / len(sub):>4.0%})'
              + ('   <-- above average' if m / len(sub) > overall + 0.1 else ''))
    print(f'  {"(all gulps)":<56} {n:>4} gulps, missed {len(miss):>3} ({overall:>4.0%})')

    print('\nmedians, caught vs missed:')
    for k in ['rise_db', 'speech_before', 'first100_share', 'peak_ms', 'harm100', 'duration_ms']:
        a = np.median([r[k] for r in rows if r['caught']])
        b = np.median([r[k] for r in miss])
        print(f'  {k:<16} caught {a:7.2f}   missed {b:7.2f}')

    finite = [r for r in miss if np.isfinite(r['score'])]
    gap = np.array([eta - r['score'] for r in finite])
    print(f'\nhow far below the threshold the missed onsets score (log-odds): median {np.median(gap):.1f}, '
          f'within 2 of it: {(gap <= 2).sum()}/{len(gap)}')


if __name__ == '__main__':
    main()
