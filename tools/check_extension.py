"""Testbench: does the extension's JavaScript engine decide exactly like the Python engine?

Runs extension/engine.js under Node and tools/live.py's Engine over the same lecture, both in
blocks, and compares every decision (anchor frame, kind, score, fire / no-fire).

Usage: .venv\\Scripts\\python tools\\check_extension.py [--lecture 005]
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

import gulpdata
import live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lecture', default='005')
    args = ap.parse_args()
    wav = gulpdata.wav_path(args.lecture)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'events.json')
        r = subprocess.run(['node', os.path.join(gulpdata.ROOT, 'extension', 'test', 'run_lecture.mjs'), wav, out],
                           capture_output=True, text=True, encoding='utf-8')
        print('node:', r.stdout.strip() or r.stderr.strip())
        with open(out, encoding='utf-8') as f:
            js = json.load(f)

    with open(os.path.join(gulpdata.ROOT, 'extension', 'model.json'), encoding='utf-8') as f:
        model = json.load(f)
    x16, fs = gulpdata.read_wav(args.lecture)
    eng = live.Engine(model, '2', 30.0, 1)
    xs = np.concatenate([x16.astype(np.float32) / 32768, np.zeros(eng.delay + 480, np.float32)])[:, None]
    for a in range(0, len(xs), 480):
        eng.process(xs[a:a + 480])

    key = lambda t, k: (int(round(t / live.HOP_S)), int(k))
    py = {key(t, k): (z, f) for t, k, z, f in eng.events}
    jj = {key(e[0], e[1]): (e[2], e[3]) for e in js['events']}
    both = py.keys() & jj.keys()
    zd = np.array([abs(py[k][0] - jj[k][0]) for k in both])
    agree = sum(py[k][1] == jj[k][1] for k in both)
    print(f'anchors: python {len(py)}, javascript {len(jj)}, in both {len(both)}')
    print(f'scores: max |z diff| {zd.max():.2e}, median {np.median(zd):.2e}')
    print(f'fire / no-fire agree on {agree}/{len(both)}; mutes python {sum(f for _, f in py.values())}, '
          f'javascript {sum(f for _, f in jj.values())}; late decisions javascript {js["late"]}')
    pm = [(m['start'], m['end'] if m['end'] is not None else m['start'] + live.MUTE_MAX_S) for m in eng.mute_log]
    jm = [tuple(m) for m in js['mutes']]
    if len(pm) == len(jm) and pm:
        d = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(pm, jm))
        print(f'mute intervals: {len(pm)} in both, max start/end difference {d * 1000:.3f} ms')
    else:
        print(f'mute intervals: python {len(pm)}, javascript {len(jm)}  <-- MISMATCH')


if __name__ == '__main__':
    main()
