"""Build the three review queues for the next round of analysis.

  hidden  the current onset classifier's (N = 100 ms) strongest onsets that nobody has listened
          to. About 75% of such onsets were gulps last time. Shown in random order with the
          score hidden, so neither rank nor score can anchor the judgement.
  missed  labelled gulps the classifier misses at 5 false alarms/h (stage-1 misses included).
          The question is WHY, answered with tags, not a new label.
  multi   gulp labels whose core splits into bursts more than 150 ms apart: one gulp or two
          sounds?

Selections start at the edges measured from the audio (output_roc.gulp_edges), so a gulp
labelled from this queue gets clean edges without dragging.

Usage: .venv\\Scripts\\python tools\\build_queues.py [--hidden 200]
"""
import argparse
import json
import os

import numpy as np

import gulpdata
import onset_classifier as oc
from output_roc import gulp_edges
from pu_correction import reviewed_mask

N_MS, LAM = 100, 100.0
FA_PER_H = 5
MULTI_GAP_MS = 150
SNAP_WINDOW_S = (0.05, 0.6)     # search span for the edges of a hidden candidate
FRAME_S = 0.025


def edges_s(total, s, e):
    """Measured edges in seconds, or the span itself if it holds no usable audio."""
    e = min(e, (len(total) - 1) * oc.HOP_S)
    try:
        k0, k1, core = gulp_edges(total, s, e)
    except (IndexError, ValueError):
        return round(s, 3), round(e, 3), np.array([], int)
    return round(k0 * oc.HOP_S, 3), round(k1 * oc.HOP_S + FRAME_S, 3), core


def write(name, items):
    with open(os.path.join(gulpdata.REVIEW, name), 'w', encoding='utf-8') as f:
        json.dump({'candidates': items}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=200)
    args = ap.parse_args()

    lectures, feats, base = oc.load_all()
    scores = oc.cv_scores(N_MS, LAM, lectures, feats, base)
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    neg_all = np.sort(np.concatenate([scores[l][base[l][2] == 0] for l in lectures]))
    eta = neg_all[-int(FA_PER_H * hours) - 1]

    hidden, missed, multi = [], [], []
    for l in lectures:
        on, floor, y, w, gulps = base[l]
        total = feats[l]['total']
        labels = gulpdata.load_labels(l)
        unrev = np.flatnonzero((y == 0) & ~reviewed_mask(on, labels))
        for i in unrev:
            t = on[i] * oc.HOP_S
            hidden.append({'lec': l, 'onset': round(t, 3), 'score': round(float(scores[l][i]), 2),
                           'id': f'hid_{l}_{t:.3f}'})
        glabs = [x for x in labels if x[2] in gulpdata.GULP_LABELS]
        for g, (s0, e0, lab) in zip(gulps, glabs):        # same order as lecture_onsets
            s, e, core = edges_s(total, s0, e0)
            i = g['onset_index']
            z = float(scores[l][i]) if i is not None else None
            if i is None or z <= eta:
                missed.append({'lec': l, 'start': s, 'end': e, 'label': lab,
                               'onset': round(on[i] * oc.HOP_S, 3) if i is not None else None,
                               'score': round(z, 2) if z is not None else -99.0,
                               'stage1_miss': i is None, 'id': f'miss_{l}_{s0:.3f}'})
            if len(core) > 1 and (np.diff(core) * oc.HOP_S * 1000 > MULTI_GAP_MS).any():
                multi.append({'lec': l, 'start': s, 'end': e, 'label': lab, 'score': 0.0,
                              'id': f'multi_{l}_{s0:.3f}'})

    hidden.sort(key=lambda c: -c['score'])
    hidden = hidden[:args.hidden]
    for c in hidden:
        c['start'], c['end'], _ = edges_s(feats[c['lec']]['total'], c['onset'] - SNAP_WINDOW_S[0],
                                          c['onset'] + SNAP_WINDOW_S[1])
    rng = np.random.default_rng(0)
    hidden = [hidden[i] for i in rng.permutation(len(hidden))]     # blind order
    for r, c in enumerate(hidden, 1):
        c['rank'] = r
    for lst in (missed, multi):
        lst.sort(key=lambda c: (c['lec'], c['start']))
        for r, c in enumerate(lst, 1):
            c['rank'] = r
    write('queue_hidden.json', hidden)
    write('queue_missed.json', missed)
    write('queue_multi.json', multi)
    n_g = sum(len(b[4]) for b in base.values())
    print(f'hidden: {len(hidden)} unreviewed onsets (scores {min(c["score"] for c in hidden):.1f} to '
          f'{max(c["score"] for c in hidden):.1f}), shuffled')
    print(f'missed: {len(missed)} of {n_g} gulps at {FA_PER_H} FA/h '
          f'({sum(c["stage1_miss"] for c in missed)} never found by the onset detector)')
    print(f'multi:  {len(multi)} multi-burst gulp labels')


if __name__ == '__main__':
    main()
