"""Plot results/output_roc.json (from output_roc.py) as results/output_roc.png.

One curve per lookahead L, each with the hysteresis width that does best at 5 cuts/h. That
width is picked on the same data, so the curves are slightly optimistic.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import gulpdata
from plot_latency_roc import BASE, GRID, INK, INK2, SURFACE

RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']   # sequential blue, 250/350/450/550/700


def curve(c):
    x, y = np.array(c['cuts_per_h']), np.array(c['supp'])
    o = np.argsort(x, kind='stable')
    return x[o], np.maximum.accumulate(y[o])    # best suppression reachable at each cut rate


def at(c, t):
    x, y = curve(c)
    return y[x <= t].max() if (x <= t).any() else 0.0


def main():
    out = os.path.join(gulpdata.ROOT, 'results')
    with open(os.path.join(out, 'output_roc.json'), encoding='utf-8') as f:
        d = json.load(f)
    best = {}
    for c in d['curves']:
        if c['baseline']:
            continue
        if at(c, 5) > (at(best[c['L_ms']], 5) if c['L_ms'] in best else -1):
            best[c['L_ms']] = c
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    lo, hi = 0.5, 300
    for color, L in zip(RAMP, sorted(best)):
        c = best[L]
        x, y = curve(c)
        ax.step(np.clip(x, lo, None), y, where='post', color=color, lw=2,
                label=f'L = {L} ms (hysteresis {c["delta"]:g})')
    for c in d['curves']:
        if c['baseline']:
            x, y = curve(c)
            ax.step(np.clip(x, lo, None), y, where='post', color=BASE, lw=2, ls='--', label='level only, L = 100 ms')
    for t in [1, 5, 20]:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.02)
    ticks = [1, 5, 20, 100]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel('Speech cuts per hour of lecture (log scale)', color=INK2)
    ax.set_ylabel('Gulps suppressed (≥ 90% of energy muted)', color=INK2)
    ax.set_title(f'Suppression vs speech cuts by lookahead L ({d["n_gulps"]} gulps, {d["hours"]:.1f} h, '
                 'lecture-grouped CV)', color=INK, fontsize=10, loc='left')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis='y', color=GRID, lw=1)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper left')
    fig.tight_layout()
    path = os.path.join(out, 'output_roc.png')
    fig.savefig(path, facecolor=SURFACE)
    print('saved', os.path.relpath(path, gulpdata.ROOT))


if __name__ == '__main__':
    main()
