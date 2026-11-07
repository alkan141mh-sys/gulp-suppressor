"""The 10 confirmed false alarms that are hardest for the detector: audit alarms the user
judged not a gulp that survive the read-ahead speech veto, ranked by the detector's score
(retrained on today's labels, cross-validated, so no lecture scores itself).

Rewrites data/review/queue_fa_known.json (source 'fa_known') with those 10, hardest first.

Usage: .venv\\Scripts\\python tools\\build_fa_hard.py
"""
import json
import os

import numpy as np

import gulpdata as g
import onset_classifier as oc
import reanchor as ra

N = 10
VETO_FRAMES, VETO_DB = 20, 43


def main():
    K = json.load(open(os.path.join(g.REVIEW, 'audit_key.json'), encoding='utf-8'))
    d = g.load_decisions()
    ids = dict(l.split()[:2] for l in open(os.path.join(g.ROOT, 'data', 'playlist_ids.txt'), encoding='utf-8') if l.strip())
    lectures, feats, _ = oc.load_all()
    D = {}
    for l in lectures:
        D[l] = ra.lecture_data(feats[l], g.load_labels(l))
        D[l]['total'] = feats[l]['total']
    scores = ra.cv(D, lectures, False)
    rows = []
    for cid, kind in K['key'].items():
        lab = d.get(cid, {}).get('label')
        if kind != 'alarm' or lab is None or lab in g.GULP_LABELS or lab == 'unsure':
            continue
        _, l, t = cid.split('_')
        t = float(t)
        dl, z = D[l], scores[l]
        tot = dl['total']
        on, floor = oc.detect_onsets(tot)
        k = int(round(t / oc.HOP_S))
        gi = np.flatnonzero(on == k)
        if not len(gi):
            continue
        sel = np.flatnonzero(dl['grp'] == gi[0])
        A = dl['A'][sel]
        idx = np.clip(A[:, None] + np.arange(VETO_FRAMES)[None, :], 0, len(tot) - 1)
        pk = tot[idx].max(1) - floor[A]
        zz = np.where(pk > VETO_DB, -np.inf, z[sel])
        if not np.isfinite(zz).any():
            continue                                     # vetoed: the speech veto already handles it
        q = d[cid]
        rows.append({'lec': l, 'start': q['start'], 'end': q['end'], 'onset': t, 'label': lab,
                     'score': round(float(zz.max()), 2), 'id': cid,
                     'peak_db': round(float(tot[k:k + 10].max() - floor[k]), 1)})
    rows.sort(key=lambda c: -c['score'])
    rows = rows[:N]
    for r, c in enumerate(rows, 1):
        c['rank'] = r
    with open(os.path.join(g.REVIEW, 'queue_fa_known.json'), 'w', encoding='utf-8') as f:
        json.dump({'candidates': rows}, f, indent=1)
    for c in rows:
        m, s = divmod(c['onset'], 60)
        print(f"Y{c['rank']:<3} {c['lec']}  {int(m)}:{s:06.3f}  {c['label']:<7} {c['peak_db']:>5.1f} dB  z {c['score']:>6.2f}  "
              f"https://www.youtube.com/watch?v={ids[c['lec']]}&t={max(0, int(c['onset']) - 2)}s")


if __name__ == '__main__':
    main()
