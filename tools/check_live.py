"""Testbench: does the streaming engine (live.py) compute what the batch analysis computes?

Runs one training lecture through Engine block by block and compares, stage by stage:
  1. resampler output vs scipy's resample_poly
  2. per-frame features vs the batch feature cache
  3. anchor positions and kinds vs reanchor.py
  4. scores vs the same final model applied in batch (the only intended difference is the
     speech-level reference: batch uses the whole lecture, live a running 60 s estimate)
  5. timing: no mute decided after its start had already been played

Usage: .venv\\Scripts\\python tools\\check_live.py [--lecture 005]
"""
import argparse
import json

import numpy as np
from scipy.signal import resample_poly

import features
import gulpdata
import live
import reanchor as ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lecture', default='005')
    args = ap.parse_args()
    lec = args.lecture
    with open(live.MODEL, encoding='utf-8') as f:
        model = json.load(f)

    x16, fs = gulpdata.read_wav(lec)
    x = x16.astype(np.float32) / 32768

    # 1. Resampler, fed in 480-sample blocks.
    rs = live.Resampler()
    y = np.concatenate([rs.push(x[a:a + 480].astype(np.float64)) for a in range(0, len(x), 480)])
    ref = resample_poly(x, features.FS, fs)
    n = min(len(y), len(ref)) - 100
    print(f'1. resampler: max |live - resample_poly| = {np.abs(y[:n] - ref[:n]).max():.2e} over {n} samples')

    # 2-5. Engine on the whole lecture.
    eng = live.Engine(model, '2', 30.0, 1)
    eng.trace = {'bands': [], 'total': []}
    xs = np.concatenate([x, np.zeros(eng.delay + 480, np.float32)])[:, None]
    for a in range(0, len(xs), 480):
        eng.process(xs[a:a + 480])
    f = features.lecture_features(lec)
    T = min(len(eng.trace['total']), len(f['total']))
    dt = np.abs(np.array(eng.trace['total'][:T]) - f['total'][:T])
    db = np.abs(np.array(eng.trace['bands'][:T]) - f['bands'][:T])
    live_bands = np.array(eng.trace['bands'][:T])
    audible = (f['bands'][:T] > -100) & (live_bands > -100)    # bins above digital silence
    print(f'2. features over {T} frames: max |total diff| {dt.max():.2e} dB, max |band diff| {db.max():.2e} dB, '
          f'in bins above -100 dB {db[audible].max():.2e} dB (float32 rounding in the batch path is ~1e-5)')

    d = ra.lecture_data(f, gulpdata.load_labels(lec))
    batch = {(int(k), int(kd)) for k, kd in zip(d['A'], d['kind'])}
    stream = {(int(round(t / live.HOP_S)), int(kd)) for t, kd, z, fd in eng.events}
    print(f'3. anchors: batch {len(batch)}, live {len(stream)}, in both {len(batch & stream)}, '
          f'only batch {len(batch - stream)}, only live {len(stream - batch)}')

    mu, sd, beta = np.array(model['mu']), np.array(model['sd']), np.array(model['beta'])
    zb = np.clip((d['X'] - mu) / sd, -4, 4) @ beta[:-1] + beta[-1]
    zbatch = {(int(k), int(kd)): z for k, kd, z in zip(d['A'], d['kind'], zb)}
    eta = model['thresholds_by_fa_per_h']['2']
    pairs = [(zbatch[(int(round(t / live.HOP_S)), int(kd))], z, fd) for t, kd, z, fd in eng.events
             if (int(round(t / live.HOP_S)), int(kd)) in zbatch]
    zd = np.array([abs(a - b) for a, b, _ in pairs])
    agree = np.mean([(a > eta) == fd for a, _, fd in pairs])
    late60 = [abs(a - b) for (a, b, _), (t, *_r) in zip(pairs, eng.events) if t > 60]
    print(f'4. scores: median |z diff| {np.median(zd):.3f}, after the first minute {np.median(late60):.3f}; '
          f'fire / no-fire decisions agree on {agree:.1%} of {len(pairs)} anchors')
    print(f'5. timing: {eng.late} late decisions (mute started after its audio was already played)')

    fired = [t for t, kd, z, fd in eng.events if fd]
    gl = [(s, e) for s, e, lab in gulpdata.load_labels(lec) if lab in gulpdata.GULP_LABELS]
    hit = sum(any(s - 0.05 <= t <= e for t in fired) for s, e in gl)
    print(f'   lecture {lec}: {len(fired)} mutes in {len(x) / fs / 60:.1f} min; {hit}/{len(gl)} labelled gulps muted '
          '(in-sample: this lecture trained the final model, so this checks plumbing, not accuracy)')


if __name__ == '__main__':
    main()
