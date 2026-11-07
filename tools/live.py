"""Real-time gulp suppressor (software version), plus an offline mode with the same engine.

Signal path, every 10 ms:
  48 kHz in -> FIR low-pass + decimate by 3 (the exact filter of scipy's resample_poly, so the
  features match training) -> 25 ms frame stats (features.frame_stats) -> floor tracker ->
  onset detector + secondary anchors -> 100 ms patch -> logistic score -> mute schedule.
  The audio itself goes through a DELAY_MS delay line and a gain with 5 ms ramps.

Why DELAY_MS = 115: an anchor's decision needs the frames up to 95.6 ms after it (8 frames of
10 ms, the last one 25 ms long, plus the resampler's 0.6 ms). Audio arrives in 10 ms blocks, so
the decision can be up to one block later. 115 ms leaves a few ms of margin and stays under the
~125 ms where a late soundtrack becomes noticeable against the picture.

Modes:
  python tools/live.py --list                                 show audio devices
  python tools/live.py --in "CABLE Output" --out "Headphones"  run live
  python tools/live.py --file in.wav --wav-out out.wav         process a file (offline check)

Live keys: G = a gulp got through, K = something that is not a gulp was cut, B = bypass on/off,
Q = quit. G and K save the last REPORT_S seconds (input, output, decisions) to data/live_reports.
"""
import argparse
import datetime
import json
import os
import sys
import threading
import time
import wave

import numpy as np
from scipy.signal import firwin

import features
import gulpdata

FS_IN = 48000
DOWN = FS_IN // features.FS            # 3
HALF = 10 * DOWN                        # resample_poly's default half length
H = firwin(2 * HALF + 1, 1 / DOWN, window=('kaiser', 5.0))
HOP_S = features.HOP / features.FS
FRAME_S = features.WIN / features.FS
DELAY_MS = 115
RAMP_MS = 5
MUTE_END_QUIET = 3                      # frames back near the floor that end a mute
MUTE_MAX_S = 0.5                        # longest measured gulp: 510 ms
# A mute also ends when speech resumes: SPEECH_FRAMES frames in a row that are voiced
# (harmonicity >= SPEECH_HARM) and loud (>= floor + SPEECH_DB). Measured on the 153 labelled
# gulps: this fires inside only 6 of them, but finds speech starting right after 76 of them
# (median 210 ms later), where a mute used to run on into the next word.
SPEECH_HARM = 0.8
SPEECH_DB = 25.0
SPEECH_FRAMES = 2
RING = 8192                             # frames of history kept (82 s)
P95_FRAMES = 6000                       # speech-level reference over the last 60 s
REPORT_S = 8
MODEL = os.path.join(gulpdata.ROOT, 'models', 'detector.json')
REPORTS = os.path.join(gulpdata.ROOT, 'data', 'live_reports')


class Resampler:
    """Streaming 48 -> 16 kHz that reproduces resample_poly(x, 1, 3) sample for sample.

    resample_poly centres its 61-tap filter: y[n] = sum_j H[3n + 30 - j] x[j], x = 0 before
    the start. Output n therefore needs input up to 3n + 30 (0.6 ms of lookahead).
    """

    def __init__(self):
        self.buf = np.zeros(HALF, np.float64)   # zeros stand for x[-30..-1]
        self.start = -HALF                       # global index of buf[0]
        self.n = 0                               # next output index

    def push(self, x):
        self.buf = np.concatenate([self.buf, x])
        last = self.start + len(self.buf) - 1    # newest input index available
        n_end = (last - HALF) // DOWN            # last computable output
        if n_end < self.n:
            return np.zeros(0)
        ns = np.arange(self.n, n_end + 1)
        lo = DOWN * ns - HALF - self.start       # buf position of x[3n - 30]
        win = np.lib.stride_tricks.sliding_window_view(self.buf, 2 * HALF + 1)
        y = win[lo] @ H[::-1]
        self.n = n_end + 1
        keep = DOWN * self.n - HALF - self.start # first input still needed
        self.buf = self.buf[keep:]
        self.start += keep
        return y


class Engine:
    def __init__(self, model, fa_per_h='2', atten_db=30.0, channels=1):
        p = model['params']
        self.p = p
        self.mu, self.sd = np.array(model['mu']), np.array(model['sd'])
        self.beta = np.array(model['beta'])
        th = model['thresholds_by_fa_per_h']
        self.eta = th.get(str(fa_per_h), th[model.get('default_fa', next(iter(th)))])
        self.p95 = model['p95_default']
        self.g_mute = 10 ** (-atten_db / 20)
        self.delay = int(DELAY_MS / 1000 * FS_IN)
        self.step = 1 / (RAMP_MS / 1000 * FS_IN)
        self.rs = Resampler()
        self.y16 = np.zeros(0)
        self.y_off = 0                           # 16 kHz index of y16[0]
        self.k = 0                               # next frame
        self.bands = np.zeros((RING, features.N_BANDS))
        self.flux = np.zeros(RING)
        self.total = np.zeros(RING)
        self.harm = np.zeros(RING)
        self.sm = np.zeros(RING)
        self.floor = np.zeros(RING)
        self.above = np.zeros(RING, bool)
        self.dig = np.zeros(RING, bool)
        self.sounds, self.pending, self.mutes = [], [], []
        self.fifo = np.zeros((self.delay, channels), np.float32)
        self.s_in = 0
        self.s_in_block = 0                      # input sample count at the start of this block
        self.gain = 1.0
        self.bypass = False
        self.events = []                         # (input time s, kind, z, fired)
        self.mute_log = []                       # every mute ever scheduled (for evaluation)
        self.late = 0
        self.lock = threading.Lock()
        self.trace = None                        # check_live.py sets a dict to record every frame

    # ---- detector, one 10 ms frame at a time ----
    def _frame(self, fr):
        k, p, R = self.k, self.p, RING
        b, t, h, _ = features.frame_stats(fr[None, :])
        i = k % R
        self.bands[i], self.total[i], self.harm[i] = b[0], t[0], h[0]
        if self.trace is not None:
            self.trace['bands'].append(b[0])
            self.trace['total'].append(t[0])
        self.flux[i] = np.maximum(b[0] - self.bands[(k - 1) % R], 0).sum() if k else 0.0
        n5 = min(5, k + 1)
        self.sm[i] = self.total[[(k - j) % R for j in range(n5)]].mean()
        fl_n = p['FLOOR_N']
        if k == 0:
            self.floor[i] = self.sm[i]
        else:
            idx = [(k - j) % R for j in range(1, min(fl_n, k) + 1)]
            self.floor[i] = self.sm[idx].min()
        self.above[i] = t[0] >= self.floor[i] + p['ONSET_DB']
        self.dig[i] = t[0] <= p['DIGITAL_SILENCE_DB']
        q = int(p['QUIET_MS'] / 1000 / HOP_S)
        quiet_before = not any(self.above[(k - j) % R] for j in range(1, min(q, k) + 1))
        enabled = k >= 2 * fl_n and not any(self.dig[(k - j) % R] for j in range(0, 2 * fl_n + 1))
        if k % 100 == 0 and k >= P95_FRAMES:
            last = self.total[[(k - j) % R for j in range(P95_FRAMES)]]
            self.p95 = float(np.percentile(last, 95))

        # Secondary anchors inside sounds that are still open (same rules as reanchor.py).
        for s in list(self.sounds):
            if k > s['k0'] + p['MAX_AFTER']:
                self.sounds.remove(s)
                continue
            if t[0] < s['floor'] + p['ONSET_DB']:
                s['quiet'] += 1
                if s['quiet'] >= p['END_QUIET']:
                    self.sounds.remove(s)
                    continue
            else:
                s['quiet'] = 0
            weak = p['LEADIN_DB'] is None or s['top'] < s['floor'] + p['LEADIN_DB']
            if weak and t[0] >= s['top'] + p['JUMP_DB'] and k - s['last'] >= p['MIN_GAP'] and s['n2'] < p['MAX_SECONDARY']:
                self.pending.append({'k': k, 'kind': 1, 'k0': s['k0']})
                s['last'], s['n2'] = k, s['n2'] + 1
            s['top'] = max(s['top'], t[0])
        if self.above[i] and quiet_before and enabled:
            self.sounds.append({'k0': k, 'floor': self.floor[i], 'top': t[0], 'last': k, 'quiet': 0, 'n2': 0})
            self.pending.append({'k': k, 'kind': 0, 'k0': k})

        # Decisions whose 100 ms of audio is now complete.
        for a in [a for a in self.pending if a['k'] + p['n_after'] - 1 <= k]:
            self.pending.remove(a)
            self._decide(a)
        # Open mutes end when the sound is back near its floor, when speech resumes, or at the cap.
        for m in self.mutes:
            if m['end'] is None and m['seen'] < k:
                m['seen'] = k
                self._mute_step(m, k)
        self.k += 1

    def _mute_step(self, m, j):
        """Advance an open mute by frame j; may set m['end']."""
        R, p = RING, self.p
        t, h = self.total[j % R], self.harm[j % R]
        if t < m['floor'] + p['ONSET_DB']:
            m['quiet'] += 1
            if m['quiet'] >= MUTE_END_QUIET:
                m['end'] = (j - MUTE_END_QUIET + 1) * HOP_S + FRAME_S
        else:
            m['quiet'] = 0
        if j > m['k'] and h >= SPEECH_HARM and t >= m['floor'] + SPEECH_DB:
            m['speech'] += 1
            if m['speech'] >= SPEECH_FRAMES and m['end'] is None:
                m['end'] = max(m['start'], (j - SPEECH_FRAMES + 1) * HOP_S - HOP_S)   # just before the speech
        else:
            m['speech'] = 0
        if m['end'] is None and (j * HOP_S + FRAME_S) - m['start'] >= MUTE_MAX_S:
            m['end'] = m['start'] + MUTE_MAX_S

    def _decide(self, a):
        p, R = self.p, RING
        ka = a['k']
        idx = [(min(max(ka + j, 0), self.k) % R) for j in range(-p['PRE'], p['n_after'])]
        fl = self.floor[ka % R]
        per = np.column_stack([self.bands[idx] - fl, self.harm[idx], self.flux[idx]])
        x = np.concatenate([per.ravel(), [fl - self.p95, a['kind'], (ka - a['k0']) * HOP_S * 1000]])
        z = float(np.clip((x - self.mu) / self.sd, -4, 4) @ self.beta[:-1] + self.beta[-1])
        fired = z > self.eta
        t0 = ka * HOP_S
        self.events.append((t0, a['kind'], z, fired))
        if not fired:
            return
        if t0 < (self.s_in_block - self.delay) / FS_IN:
            self.late += 1                     # too late to catch the very start
        open_m = next((m for m in self.mutes if m['end'] is None or m['end'] > t0), None)
        if open_m and open_m['start'] <= t0:
            open_m['quiet'] = 0
            if open_m['end'] is not None and open_m['end'] < t0 + FRAME_S:
                open_m['end'] = None           # extend: still sounding
                open_m['seen'] = self.k
            return
        m = {'start': t0, 'end': None, 'floor': fl, 'quiet': 0, 'speech': 0, 'k': ka, 'seen': self.k}
        # The sound may already be over by now (the decision comes 100 ms late): check history.
        for j in range(ka, self.k + 1):
            self._mute_step(m, j)
            if m['end'] is not None:
                break
        self.mutes.append(m)
        self.mute_log.append(m)

    # ---- audio ----
    def process(self, block):
        """block: (n, channels) float32 at 48 kHz. Returns the delayed, gain-controlled block."""
        with self.lock:
            n = len(block)
            self.s_in_block = self.s_in
            y = self.rs.push(block.mean(axis=1).astype(np.float64))
            self.y16 = np.concatenate([self.y16, y])
            W, Hh = features.WIN, features.HOP
            while (self.k * Hh + W) - self.y_off <= len(self.y16):
                a = self.k * Hh - self.y_off
                self._frame(self.y16[a:a + W].astype(np.float32))
            drop = self.k * Hh - self.y_off
            if drop > 0:
                self.y16 = self.y16[drop:]
                self.y_off += drop
            self.s_in += n
            self.fifo = np.concatenate([self.fifo, block])
            out, self.fifo = self.fifo[:n], self.fifo[n:]
            tau = (self.s_in_block - self.delay + np.arange(n)) / FS_IN   # input time of each output sample
            target = np.ones(n)
            for m in self.mutes:
                end = m['end'] if m['end'] is not None else np.inf
                target[(tau >= m['start']) & (tau < end)] = self.g_mute
            if self.bypass:
                target[:] = 1.0
            g = np.empty(n)
            cur = self.gain
            for j in range(n):                   # 5 ms linear ramps, no clicks
                d = target[j] - cur
                cur += max(-self.step, min(self.step, d))
                g[j] = cur
            self.gain = cur
            self.mutes = [m for m in self.mutes if m['end'] is None or m['end'] > tau[-1] - 1.0]
            return (out * g[:, None].astype(np.float32)), g


def run_file(engine, path_in, path_out):
    with wave.open(path_in, 'rb') as w:
        assert w.getframerate() == FS_IN and w.getsampwidth() == 2, 'expected 48 kHz 16-bit'
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, ch).astype(np.float32) / 32768
    x = np.concatenate([x, np.zeros((engine.delay + 480, ch), np.float32)])   # flush the delay line
    outs, gains = [], []
    t = time.perf_counter()
    for a in range(0, len(x), 480):
        o, g = engine.process(x[a:a + 480])
        outs.append(o)
        gains.append(g)
    dt = time.perf_counter() - t
    y = np.concatenate(outs)[engine.delay:engine.delay + len(x) - engine.delay - 480]
    if path_out:
        with wave.open(path_out, 'wb') as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(FS_IN)
            w.writeframes((np.clip(y, -1, 1 - 2 ** -15) * 32768).astype('<i2').tobytes())
    dur = (len(x) - engine.delay - 480) / FS_IN
    return dur, dt, np.concatenate(gains)


def find_device(sd, name, kind):
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d['hostapi'])['name']
        if name.lower() in d['name'].lower() and d[f'max_{kind}_channels'] > 0 and 'WASAPI' in api:
            return i
    sys.exit(f'no WASAPI {kind} device matching "{name}" (see --list)')


def save_report(engine, ring_in, ring_out, kind):
    os.makedirs(REPORTS, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    base = os.path.join(REPORTS, f'{stamp}_{kind}')
    for name, r in [('in', ring_in), ('out', ring_out)]:
        with wave.open(f'{base}_{name}.wav', 'wb') as w:
            w.setnchannels(r.shape[1])
            w.setsampwidth(2)
            w.setframerate(FS_IN)
            w.writeframes((np.clip(r, -1, 1 - 2 ** -15) * 32768).astype('<i2').tobytes())
    now = engine.s_in / FS_IN
    ev = [{'t_before_key_s': round(now - t, 3), 'secondary': bool(k), 'z': round(z, 2), 'fired': bool(f)}
          for t, k, z, f in engine.events if now - t <= REPORT_S]
    with open(f'{base}.json', 'w', encoding='utf-8') as f:
        json.dump({'kind': kind, 'threshold': engine.eta, 'events': ev}, f, indent=1)
    return base


def run_live(engine, dev_in, dev_out, ch):
    import msvcrt
    import sounddevice as sd
    i_in, i_out = find_device(sd, dev_in, 'input'), find_device(sd, dev_out, 'output')
    n_ring = REPORT_S * FS_IN
    ring_in = np.zeros((n_ring, ch), np.float32)
    ring_out = np.zeros((n_ring, ch), np.float32)
    state = {'pos': 0, 'xruns': 0, 'max_ms': 0.0}

    def cb(indata, outdata, frames, time_info, status):
        if status:
            state['xruns'] += 1
        t = time.perf_counter()
        out, _ = engine.process(indata.copy())
        outdata[:] = out
        for r, v in [(ring_in, indata), (ring_out, out)]:
            r[:-frames] = r[frames:]
            r[-frames:] = v
        state['max_ms'] = max(state['max_ms'], (time.perf_counter() - t) * 1000)

    # auto_convert lets Windows resample when a device runs at another rate (Bluetooth
    # headphones often use 44.1 kHz); the engine itself always works at 48 kHz.
    conv = sd.WasapiSettings(auto_convert=True)
    with sd.Stream(device=(i_in, i_out), samplerate=FS_IN, blocksize=480, channels=ch,
                   dtype='float32', latency='low', callback=cb, extra_settings=(conv, conv)):
        print(f'running: {sd.query_devices(i_in)["name"]} -> {sd.query_devices(i_out)["name"]}, '
              f'delay {DELAY_MS} ms. Keys: G gulp got through, K wrong cut, B bypass, Q quit')
        while True:
            time.sleep(0.05)
            if msvcrt.kbhit():
                c = msvcrt.getwch().lower()
                if c == 'q':
                    break
                if c == 'b':
                    engine.bypass = not engine.bypass
                    print('bypass', 'ON (original sound)' if engine.bypass else 'OFF (suppressing)')
                if c in 'gk':
                    with engine.lock:
                        base = save_report(engine, ring_in.copy(), ring_out.copy(), 'missed_gulp' if c == 'g' else 'wrong_cut')
                    print('saved', os.path.relpath(base, gulpdata.ROOT))
            n_mutes = sum(1 for e in engine.events if e[3])
            print(f'\r{engine.s_in / FS_IN / 60:6.1f} min  mutes {n_mutes:4d}  late {engine.late}  '
                  f'dropouts {state["xruns"]}  worst block {state["max_ms"]:.1f} ms   ', end='', flush=True)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--in', dest='dev_in', default='CABLE Output')
    ap.add_argument('--out', dest='dev_out')
    ap.add_argument('--channels', type=int, default=2)
    ap.add_argument('--file')
    ap.add_argument('--wav-out')
    ap.add_argument('--fa', default='2', help='operating point, true false alarms per hour (read-ahead): 0.5, 1, 2, 3 or 5')
    ap.add_argument('--atten', type=float, default=30.0, help='attenuation during a mute, dB')
    args = ap.parse_args()
    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return
    with open(MODEL, encoding='utf-8') as f:
        model = json.load(f)
    if args.file:
        with wave.open(args.file) as w:
            ch = w.getnchannels()
        eng = Engine(model, args.fa, args.atten, ch)
        dur, dt, g = run_file(eng, args.file, args.wav_out)
        fired = [e for e in eng.events if e[3]]
        print(f'{dur / 60:.1f} min processed in {dt:.1f} s ({dur / dt:.0f}x real time); '
              f'{len(fired)} mutes ({len(fired) / dur * 3600:.0f}/h), late decisions {eng.late}, '
              f'muted {np.mean(g < 0.5) * 100:.2f}% of the time')
        return
    if not args.dev_out:
        sys.exit('give --out (your headphones or speakers, see --list)')
    run_live(Engine(model, args.fa, args.atten, args.channels), args.dev_in, args.dev_out, args.channels)


if __name__ == '__main__':
    main()
