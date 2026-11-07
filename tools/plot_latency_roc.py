"""Plot results/latency_roc.json (from latency_roc.py) as results/latency_roc.png.

Lookahead is ordered, so the curves use one blue ramp, light (short L) to dark (long L);
the energy-only baseline is a neutral dashed line.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import gulpdata

RAMP = ['#6da7ec', '#2a78d6', '#1c5cab', '#0d366b']   # sequential blue, steps 300/450/550/700
BASE = '#8a8985'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    out = os.path.join(gulpdata.ROOT, 'results')
    with open(os.path.join(out, 'latency_roc.json'), encoding='utf-8') as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    lo, hi = 0.5, 1000
    ramp = iter(RAMP)
    for c in d['curves']:
        fa, pd = np.array(c['fa_per_h']), np.array(c['pd'])
        o = np.argsort(fa, kind='stable')
        fa, pd = fa[o], np.maximum.accumulate(pd[o])   # best P_D achievable at each FA rate
        fa = np.clip(fa, lo, None)
        base = c['name'].startswith('energy')
        color = BASE if base else next(ramp)
        ax.step(fa, pd, where='post', color=color, lw=2, ls='--' if base else '-', label=c['name'])
        if not base:
            x = 600
            y = pd[min(np.searchsorted(fa, x, 'right'), len(pd)) - 1]
            ax.annotate(f'{c["L_ms"]} ms', (x, y), xytext=(-6, 0), textcoords='offset points',
                        color=INK2, fontsize=8, va='center', ha='right',
                        bbox=dict(boxstyle='round,pad=0.15', fc=SURFACE, ec='none'))
    for t in [1, 5, 20]:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_xscale('log')
    ticks = [1, 5, 20, 100, 1000]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False alarms per hour of H0 audio (log scale)', color=INK2)
    ax.set_ylabel('Gulps detected in time, $P_D$', color=INK2)
    ax.set_title(f'Detection vs lookahead L ({d["n_events"]} gulps, {d["hours_h0"]:.1f} h H0, '
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
    path = os.path.join(out, 'latency_roc.png')
    fig.savefig(path, facecolor=SURFACE)
    print('saved', os.path.relpath(path, gulpdata.ROOT))


if __name__ == '__main__':
    main()
