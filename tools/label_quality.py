"""Check every gulp label against the audio inside it, and report suspicious ones.

For each label the measured gulp (output_roc.gulp_edges) is compared with the label:
  weak      the level rises less than WEAK_DB above the background before the label
  multi     the core (half-way rise) splits into bursts more than MULTI_GAP_MS apart:
            two sounds in one label, or a gulp plus something else
  long      the measured gulp lasts longer than LONG_MS
  offset    the measured start is more than OFFSET_MS after the label start (the label
            inherits a candidate window; harmless for training, since edges are re-measured)

Usage: .venv\\Scripts\\python tools\\label_quality.py
"""
import collections

import numpy as np

import features
import gulpdata
from output_roc import gulp_edges

HOP_S = features.HOP / features.FS
WEAK_DB = 6.0
MULTI_GAP_MS = 150
LONG_MS = 700
OFFSET_MS = 100


def main():
    rows = []
    for lec in [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]:
        labs = [x for x in gulpdata.load_labels(lec) if x[2] in gulpdata.GULP_LABELS]
        if not labs:
            continue
        tot = features.lecture_features(lec)['total']
        for s, e, lab in labs:
            k0, k1, core = gulp_edges(tot, s, e)
            a = int(round(s / HOP_S))
            bg = np.median(tot[max(0, a - 50):a]) if a > 0 else tot[a]
            rise = tot[k0:k1 + 1].max() - bg
            gaps = np.diff(core) * HOP_S * 1000 if len(core) > 1 else np.array([])
            flags = []
            if rise < WEAK_DB:
                flags.append('weak')
            if (gaps > MULTI_GAP_MS).any():
                flags.append('multi')
            if (k1 - k0 + 1) * HOP_S * 1000 > LONG_MS:
                flags.append('long')
            if (k0 * HOP_S - s) * 1000 > OFFSET_MS:
                flags.append('offset')
            rows.append((lec, s, lab, rise, (k1 - k0 + 1) * HOP_S * 1000, flags))
    n = len(rows)
    cnt = collections.Counter(f for r in rows for f in r[5])
    print(f'{n} gulp labels checked')
    for f in ['weak', 'multi', 'long', 'offset']:
        print(f'  {f:<7} {cnt[f]:>4} ({cnt[f] / n:.0%})')
    serious = [r for r in rows if set(r[5]) & {'weak', 'multi', 'long'}]
    print(f'labels with a content problem (weak, multi or long): {len(serious)}')
    for lec, s, lab, rise, dur, flags in serious[:25]:
        print(f'  {lec} {int(s // 60)}:{s % 60:06.3f}  {lab:<12} rise {rise:5.1f} dB  gulp {dur:4.0f} ms  {",".join(flags)}')
    d = np.array([r[4] for r in rows])
    print(f'measured gulp duration: median {np.median(d):.0f} ms, 10-90% {np.percentile(d, 10):.0f}-{np.percentile(d, 90):.0f} ms')


if __name__ == '__main__':
    main()
