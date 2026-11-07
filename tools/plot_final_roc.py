"""Figure for the README: catch rate against TRUE false alarms per hour for the shipped detector.

Same numbers as train_final.py (cross-validated scores, read-ahead vetoes, audit-corrected false
alarms, gulp parts < 100 ms apart counted once), drawn as a curve, with the extension's five
settings marked. Writes docs/img/roc.png.

Usage: .venv\\Scripts\\python tools\\plot_final_roc.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import gulpdata
import onset_classifier as oc
import reanchor as ra
from train_final import FA_RATES, P_MID, P_TOP, vetoed

OUT = os.path.join(gulpdata.ROOT, 'docs', 'img', 'roc.png')


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {l: {**ra.lecture_data(feats[l], gulpdata.load_labels(l)), 'total': feats[l]['total']} for l in lectures}
    cvs = vetoed(D, feats, lectures, ra.cv(D, lectures, primary_only=False))
    neg, pos, lab_pos = [], [], []
    for l in lectures:
        d, z = D[l], cvs[l]
        on, _ = oc.detect_onsets(d['total'])
        labels = gulpdata.load_labels(l)
        bad = np.zeros(d['grp'].max() + 1, bool)
        bad[d['grp'][d['y'] != 0]] = True
        best = np.full(len(bad), -np.inf)
        ok = d['y'] == 0
        np.maximum.at(best, d['grp'][ok], z[ok])
        for gi in np.flatnonzero(~bad & np.isfinite(best)):
            t = on[gi] * oc.HOP_S
            neg.append((best[gi], any(s - 0.1 <= t <= e + 0.1 for s, e, _ in labels)))
        for g in ra.merged_gulps(d):
            a = g['anchors']
            pos.append(z[a].max() if len(a) else -np.inf)
            lab_pos.append(g['label'])
    neg.sort(key=lambda x: -x[0])
    pos, lab_pos = np.array(pos), np.array(lab_pos)
    n5, n20 = int(5 * hours), int(20 * hours)
    w = np.array([1.0 if lab else (1 - P_TOP if r < n5 else 1 - P_MID if r < n20 else 1.0)
                  for r, (_, lab) in enumerate(neg)])
    s = np.array([x[0] for x in neg])
    fa = np.cumsum(w) / hours
    audible = lab_pos != 'gulp_low'
    rec = np.array([np.mean(pos > t) for t in s])
    rec_aud = np.array([np.mean(pos[audible] > t) for t in s])
    keep = fa <= 8

    model = json.load(open(os.path.join(gulpdata.ROOT, 'models', 'detector.json'), encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(fa[keep], 100 * rec_aud[keep], color='#1f6f3f', lw=1.6,
            label=f'audible gulps (not gulp_low, n = {audible.sum()})')
    ax.plot(fa[keep], 100 * rec[keep], color='#2a78d6', lw=1.8, label=f'all labelled gulps (n = {len(pos)})')
    for t in FA_RATES:
        eta = model['thresholds_by_fa_per_h'][str(t)]
        c = 100 * np.mean(pos > eta)
        ax.plot([t], [c], 'o', color='#2a78d6', ms=5)
        ax.annotate(f'{c:.0f}%', (t, c), textcoords='offset points', xytext=(4, -12), fontsize=8)
    ax.set_xlim(0, 8)
    ax.set_ylim(50, 101)
    ax.set_xlabel('true false alarms per hour (audit-corrected)')
    ax.set_ylabel('gulps muted (%)')
    ax.set_title('Read-ahead detector, lecture-grouped 6-fold cross-validation', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f'saved {os.path.relpath(OUT, gulpdata.ROOT)}; {hours:.1f} h, {len(pos)} gulps, {audible.sum()} audible')
    for t in FA_RATES:
        eta = model['thresholds_by_fa_per_h'][str(t)]
        print(f'  {t:>3} FA/h: all {np.mean(pos > eta):.1%}, audible {np.mean(pos[audible] > eta):.1%}')


if __name__ == '__main__':
    main()
