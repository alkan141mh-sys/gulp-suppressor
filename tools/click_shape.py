"""Look at marked clicks (gulp_click): sample-level waveform, envelope and spectrum.

The question is whether they are mouth sounds or a mouse button. Measurable hints:
  - attack: a mechanical click rises in well under 1 ms; mouth clicks are also fast, so this
    alone does not decide it,
  - sub-events: a mouse click is a press and a release, typically 50-150 ms apart,
  - spectrum: where the energy sits,
  - timing: a physiological click should sit at a consistent delay after the gulp, while a
    mouse click should not care where the gulp is (needs many marked clicks).

Usage: .venv\\Scripts\\python tools\\click_shape.py   -> results/click_shape.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import gulpdata as g

PAD = 0.15


def main():
    d = g.load_decisions()
    clicks = [(k[6:], v) for k, v in sorted(d.items()) if v['label'] == g.CLICK_LABEL]
    rows = [(cid, v['lec'], a, b) for cid, v in clicks for a, b in g.spans_of(v)]
    fig, ax = plt.subplots(len(rows), 3, figsize=(15, 3.2 * len(rows)), squeeze=False)
    for r, (cid, lec, a, b) in enumerate(rows):
        x, fs = g.read_wav(lec, a - PAD, b + PAD)
        x = x.astype(float) / 32768
        t = np.arange(len(x)) / fs + a - PAD
        inside = (t >= a) & (t <= b)
        # 1 ms RMS envelope in dB, to see sub-events and attack time
        w = max(1, int(fs * 0.001))
        env = 10 * np.log10(np.convolve(x ** 2, np.ones(w) / w, 'same') + 1e-12)
        floor = np.percentile(env, 20)
        pk = np.argmax(np.where(inside, env, -np.inf))
        # attack: time from 10 dB below the peak to the peak
        j = pk
        while j > 0 and env[j] > env[pk] - 10:
            j -= 1
        attack_ms = (pk - j) / fs * 1000
        above = inside & (env > env[pk] - 20)
        # sub-events: separate peaks within 20 dB of the maximum, at least 15 ms apart
        cand = np.where(inside & (env > env[pk] - 12))[0]
        events = []
        for i in cand:
            if not events or i - events[-1][-1] > 0.015 * fs:
                events.append([i])
            else:
                events[-1].append(i)
        ev_t = [(t[e[0]] - a) * 1000 for e in events]
        seg = x[inside]
        f = np.fft.rfftfreq(len(seg), 1 / fs)
        P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        cen = (f * P).sum() / P.sum()
        ax[r, 0].plot(t, x, lw=.6)
        ax[r, 0].axvspan(a, b, color='pink', alpha=.3)
        ax[r, 0].set_title(f'{cid}  click {a:.3f}-{b:.3f} s')
        ax[r, 1].plot(t, env, lw=.8)
        ax[r, 1].axhline(floor, color='gray', ls=':')
        ax[r, 1].axvspan(a, b, color='pink', alpha=.3)
        ax[r, 1].set_title(f'1 ms envelope: peak {env[pk] - floor:.0f} dB over floor, attack {attack_ms:.1f} ms, '
                           f'{len(events)} sub-event(s) at {", ".join(f"{v:.0f}" for v in ev_t)} ms')
        ax[r, 2].semilogy(f, P + 1e-12, lw=.6)
        ax[r, 2].set_xlim(0, fs / 2)
        ax[r, 2].set_title(f'spectrum of the click span, centroid {cen:.0f} Hz')
        print(f'{cid}: peak {env[pk]-floor:.0f} dB over floor, attack {attack_ms:.1f} ms, '
              f'>-20 dB for {above.sum()/fs*1000:.0f} ms, sub-events at {ev_t} ms, centroid {cen:.0f} Hz')
    fig.tight_layout()
    out = os.path.join(g.ROOT, 'results', 'click_shape.png')
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == '__main__':
    main()
