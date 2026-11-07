"""What do the extension's mutes actually hit? Runs extension/engine.js (Node) over every
training lecture and measures, from the mute intervals:

  gulps >= 80%     share of labelled gulps with at least 80% of their energy muted
  voiced cut s/h   seconds per hour of VOICED speech muted outside gulps (harmonicity >= 0.8,
                   level >= floor + 25 dB): the 'swallowed the next word' problem
  active cut s/h   seconds per hour of any audible non-gulp audio muted

The final model trained on these lectures, so the gulp share is optimistic. The comparison
between mute rules is fair: both runs use the same detections, only the mute ends differ.

Usage: .venv\\Scripts\\python tools\\eval_mutes.py [--rate 1] [--wav-dir DIR] [--compare]
"""
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import features
import gulpdata
import onset_classifier as oc
from output_roc import gulp_edges

RUNNER = os.path.join(gulpdata.ROOT, 'extension', 'test', 'run_lecture.mjs')


def run_node(wav, extra):
    fd, out = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    try:
        subprocess.run(['node', RUNNER, wav, out] + extra, check=True, capture_output=True)
        with open(out, encoding='utf-8') as f:
            return json.load(f)
    finally:
        os.remove(out)


def score(lec, mutes, scale=1.0):
    """Metrics for one lecture. scale = playback rate: audio time = content time / rate."""
    f = features.lecture_features(lec)
    tot, harm = f['total'], f['harm']
    _, floor = oc.detect_onsets(tot)
    T = len(tot)
    muted = np.zeros(T, bool)
    for a, b in mutes:                       # mute times are in audio time; frames are content time
        muted[max(0, int(a * scale / 0.01)):min(T, int(np.ceil(b * scale / 0.01)))] = True
    gulp = np.zeros(T, bool)
    fr = []
    fl10 = np.percentile(tot, 10)
    for s, e, lab in gulpdata.load_labels(lec):
        if lab not in gulpdata.GULP_LABELS:
            continue
        k0, k1, _ = gulp_edges(tot, s, e)
        gulp[max(0, k0 - 10):k1 + 11] = True
        en = np.maximum(10 ** ((tot[k0:k1 + 1] - fl10) / 10) - 1, 0)
        fr.append((en * muted[k0:k1 + 1]).sum() / max(en.sum(), 1e-9))
    active = (tot >= fl10 + 10) & ~gulp
    voiced = active & (harm >= 0.8) & (tot >= floor + 25)
    return fr, (muted & voiced).sum() * 0.01, (muted & active).sum() * 0.01, T * 0.01


def evaluate(lectures, extra, wav_for, scale):
    with ThreadPoolExecutor(max_workers=6) as ex:
        runs = list(ex.map(lambda l: run_node(wav_for(l), extra), lectures))
    fr, vc, ac, secs = [], 0.0, 0.0, 0.0
    for l, r in zip(lectures, runs):
        a, b, c, d = score(l, r['mutes'], scale)
        fr += a; vc += b; ac += c; secs += d
    h = secs / 3600
    fr = np.array(fr)
    return {'gulps_80': float(np.mean(fr >= 0.8)), 'gulps_any': float(np.mean(fr > 0)),
            'voiced_cut_s_per_h': vc / h, 'active_cut_s_per_h': ac / h,
            'late': sum(r['late'] for r in runs), 'n_gulps': len(fr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rate', type=float, default=1.0, help='playback rate to tell the engine')
    ap.add_argument('--wav-dir', help='folder with sped-up WAVs named <lecture>.wav (default: data/raw)')
    ap.add_argument('--lectures', help='comma-separated subset')
    ap.add_argument('--compare', action='store_true', help='also run the old mute rule')
    ap.add_argument('--no-rate-adapt', action='store_true', help='play sped-up audio but tell the engine 1x')
    args = ap.parse_args()
    lectures = args.lectures.split(',') if args.lectures else [l for l in gulpdata.lectures() if l not in gulpdata.TEST_LECTURES]
    wav_for = (lambda l: os.path.join(args.wav_dir, f'{l}.wav')) if args.wav_dir else gulpdata.wav_path
    told = 1.0 if args.no_rate_adapt else args.rate
    base = ['--rate', str(told)]
    configs = [('new mute rule', base)] + ([('old mute rule', base + ['--old-mutes'])] if args.compare else [])
    print(f'{len(lectures)} lectures, audio at {args.rate}x, engine told {told}x')
    for name, extra in configs:
        r = evaluate(lectures, extra, wav_for, args.rate)
        print(f'  {name:<14} gulps >=80% muted {r["gulps_80"]:.0%} (any mute {r["gulps_any"]:.0%}, n={r["n_gulps"]}); '
              f'voiced speech cut {r["voiced_cut_s_per_h"]:.1f} s/h; any non-gulp audio cut {r["active_cut_s_per_h"]:.1f} s/h; '
              f'late {r["late"]}', flush=True)


if __name__ == '__main__':
    main()
