"""'Tiny sound, then speech' rule: measured on gulps and on reviewed non-gulp sounds.

For a sound starting at onset k0:
  speech_at  first frame within 400 ms where the level reaches floor + 43 dB (None if never)
  pre_lf     peak level below 3.2 kHz (dB above the floor) from k0 until speech_at - 1 frame:
             how strong the sound is BEFORE the speech arrives
Rule: speech_at is not None and pre_lf < T  ->  not a gulp.
Prints, for each T, how many gulps (merged, at their first onset) and how many reviewed
non-gulp sounds (all review labels, plus the audit alarms) the rule would remove.

Usage: .venv\\Scripts\\python tools\\click_before_speech.py
"""
import numpy as np

import features
import gulpdata as g
import onset_classifier as oc
import reanchor as ra
from breath_rule import sound_feats

LF_BANDS, W, SPEECH_DB = 19, 40, 43


def measure(f, floor, k0):
    """(speech_at ms, low-band peak of the SOUND itself, its duration ms). The sound ends after
    100 ms back near the floor (breath_rule.sound_feats); speech arriving later is not part of it."""
    tot = f['total']
    fl = floor[k0]
    seg = tot[k0:k0 + W] - fl
    hit = np.flatnonzero(seg >= SPEECH_DB)
    if not len(hit):
        return None, None, None
    x = sound_feats(f, floor, k0)
    return int(hit[0]) * 10, float(x['lf_peak']), int(x['dur'])


def main():
    G, N = [], []
    for l in g.lectures():
        if l in g.TEST_LECTURES:
            continue
        f = features.lecture_features(l)
        on, floor = oc.detect_onsets(f['total'])
        d = ra.lecture_data(f, g.load_labels(l))
        for gg in ra.merged_gulps(d):
            a = gg['anchors']
            if len(a):
                k0 = int(on[d['grp'][a[0]]])
                G.append(measure(f, floor, k0))
        t = on * oc.HOP_S
        for s, e, lab in g.load_labels(l):
            if lab in g.GULP_LABELS or lab == 'unsure':
                continue
            hit = np.flatnonzero((t >= s - 0.1) & (t <= e + 0.1))
            for k0 in on[hit[:1]]:
                N.append((l, round(k0 * oc.HOP_S, 2), lab) + measure(f, floor, int(k0)))
    gs = [x for x in G if x[0] is not None]
    print(f'{len(G)} gulps; {len(gs)} have speech (>= {SPEECH_DB} dB) within 400 ms')
    ns = [x for x in N if x[3] is not None]
    print(f'{len(N)} reviewed non-gulp sounds; {len(ns)} with speech within 400 ms\n')
    print('gulps with speech within 400 ms (speech after ms, sound low-band peak dB, sound ms):', sorted(gs, key=lambda x: x[1]))
    for T in (10, 12, 14, 16, 18):
        gl = sum(1 for x in gs if x[1] < T and x[2] < x[0])
        nr = [x for x in ns if x[4] < T and x[5] < x[3]]
        print(f'sound ends before the speech and its low-band peak < {T:>2} dB: gulps removed {gl}/{len(G)}, '
              f'reviewed non-gulps removed {len(nr)}/{len(N)}')
    print('\nreviewed non-gulps whose sound ends before the speech (lecture, time, label, speech ms, lf peak, sound ms):')
    for x in sorted([x for x in ns if x[5] < x[3]], key=lambda x: x[4]):
        print('  ', x[:3], x[3], round(x[4], 1), x[5])


if __name__ == '__main__':
    main()
