"""Every sound labelled as a gulp, in one WAV for listening, plus a numbered index.

Each clip: a short beep, 0.3 s of silence, then the gulp label's span with CONTEXT_S of audio on
both sides. Clips are in lecture / time order; results/gulps_index.csv gives each clip's number,
start time inside the compilation, lecture, time, label, where the label came from, and a
YouTube link a few seconds before it.

Usage: .venv\\Scripts\\python tools\\export_gulp_compilation.py
"""
import csv
import os
import wave

import numpy as np

import gulpdata

CONTEXT_S = 0.5
BEEP_S, BEEP_HZ, GAP_S = 0.06, 1000, 0.3
OUT_WAV = os.path.join(gulpdata.ROOT, 'results', 'gulps_all.wav')
OUT_CSV = os.path.join(gulpdata.ROOT, 'results', 'gulps_index.csv')


def main():
    ids = {}
    with open(os.path.join(gulpdata.ROOT, 'data', 'playlist_ids.txt'), encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                k, v = line.split()
                ids[k] = v
    dec = gulpdata.load_decisions()
    from_tool = {(v['lec'], round(v['start'], 3)) for v in dec.values()}

    fs = 48000
    t = np.arange(int(BEEP_S * fs)) / fs
    beep = (0.2 * np.sin(2 * np.pi * BEEP_HZ * t) * np.hanning(len(t)) * 32767).astype(np.int16)
    gap = np.zeros(int(GAP_S * fs), np.int16)
    parts, rows, pos = [], [], 0.0
    n = 0
    for lec in gulpdata.lectures():
        if lec in gulpdata.TEST_LECTURES:
            continue
        for s, e, lab in gulpdata.load_labels(lec):
            if lab not in gulpdata.GULP_LABELS:
                continue
            n += 1
            x, fs = gulpdata.read_wav(lec, max(0.0, s - CONTEXT_S), e + CONTEXT_S)
            parts += [beep, gap, x, gap]
            rows.append({'no': n, 'at_in_file': f'{int(pos // 60)}:{pos % 60:04.1f}', 'lecture': lec,
                         'time': f'{int(s // 60)}:{s % 60:06.3f}', 'label': lab,
                         'source': 'review tool' if (lec, round(s, 3)) in from_tool else 'Audacity',
                         'link': f'https://www.youtube.com/watch?v={ids[lec]}&t={max(0, int(s) - 3)}s'})
            pos += (len(beep) + 2 * len(gap) + len(x)) / fs
    y = np.concatenate(parts)
    with wave.open(OUT_WAV, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(y.tobytes())
    with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f'{n} gulps, {len(y) / fs / 60:.1f} min -> {os.path.relpath(OUT_WAV, gulpdata.ROOT)}, '
          f'{os.path.relpath(OUT_CSV, gulpdata.ROOT)}')


if __name__ == '__main__':
    main()
