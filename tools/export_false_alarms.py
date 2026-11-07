"""Put the detector's strongest false alarms into the review tool.

Each one is either a gulp nobody labelled (it then fixes P_D and the false-alarm count) or an
H0 sound that fools the detector (a hard negative to train on). Scores come from the
cross-validated L = 200 ms detector of latency_roc.py, so no lecture is scored by a model
trained on it. Writes data/review/candidates_fa.json; candidates.json is left alone.

A 'false alarm' here is one hold-merged firing region in H0 audio; its review span is the
detector's trailing window at the peak (the audio it had heard when it fired).

Usage: .venv\\Scripts\\python tools\\export_false_alarms.py [--count 150]
"""
import argparse
import json
import os

import numpy as np

import features
import gulpdata
import latency_roc as lr

PAD_S = 0.1   # extra span shown before the trailing window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--lookahead', type=int, default=200, help='detector lookahead in ms')
    args = ap.parse_args()

    lectures = [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]
    feats = {l: features.lecture_features(l) for l in lectures}
    labels = {l: gulpdata.load_labels(l) for l in lectures}
    r = lr.run(args.lookahead, lectures, feats, labels)

    # Spans already listened to (labels, decisions) or queued (unreviewed ranker candidates).
    known = {l: [(s, e) for s, e, _ in labels[l]] for l in lectures}
    with open(os.path.join(gulpdata.REVIEW, 'candidates.json'), encoding='utf-8') as f:
        for c in json.load(f)['candidates']:
            if c['lec'] in known:
                known[c['lec']].append((c['start'], c['end']))

    hold = int(lr.HOLD_MS / 1000 / lr.HOP_S)
    win_s = (lr.n_frames(args.lookahead) - 1) * lr.HOP_S + lr.FRAME_S
    peaks = []
    for lec in lectures:
        s = np.where(r['h0'][lec], r['scored'][lec], -np.inf)
        idx = np.flatnonzero(np.isfinite(s) & (s > 0))
        if not len(idx):
            continue
        for g in np.split(idx, np.flatnonzero(np.diff(idx) > hold) + 1):
            j = g[np.argmax(s[g])]
            t1 = j * lr.HOP_S + lr.FRAME_S           # end of the audio heard at the peak
            t0 = max(0.0, t1 - win_s - PAD_S)
            if any(t0 < b and a < t1 for a, b in known[lec]):
                continue
            peaks.append({'lec': lec, 'start': round(t0, 3), 'end': round(t1, 3), 'score': round(float(s[j]), 2)})
    peaks.sort(key=lambda p: -p['score'])
    peaks = peaks[:args.count]
    for i, p in enumerate(peaks, 1):
        p['id'] = f"fa{args.lookahead}_{p['lec']}_{p['start']:.3f}"
        p['rank'] = i
        p['source'] = 'false_alarm'

    out = os.path.join(gulpdata.REVIEW, 'candidates_fa.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'model': {'lookahead_ms': args.lookahead, 'hold_ms': lr.HOLD_MS}, 'candidates': peaks}, f, indent=1)
    per_h = len(peaks) / r['hours']
    print(f'{len(peaks)} false alarms written to {os.path.relpath(out, gulpdata.ROOT)} '
          f'(strongest {per_h:.1f}/h of {r["hours"]:.1f} h; score range {peaks[-1]["score"]:.1f}-{peaks[0]["score"]:.1f})')


if __name__ == '__main__':
    main()
