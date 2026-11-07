"""The detector's confirmed false alarms: audit alarms the user labelled as not a gulp.
Writes data/review/queue_fa_known.json (source 'fa_known' in the review tool, not blind) and
prints them with YouTube links.

Usage: .venv\\Scripts\\python tools\\build_fa_known.py
"""
import json
import os

import features
import gulpdata as g
import onset_classifier as oc


def main():
    K = json.load(open(os.path.join(g.REVIEW, 'audit_key.json'), encoding='utf-8'))
    d = g.load_decisions()
    ids = dict(l.split()[:2] for l in open(os.path.join(g.ROOT, 'data', 'playlist_ids.txt'), encoding='utf-8') if l.strip())
    items = []
    for cid, kind in K['key'].items():
        lab = d.get(cid, {}).get('label')
        if kind != 'alarm' or lab is None or lab in g.GULP_LABELS or lab == 'unsure':
            continue
        q = d[cid]
        _, lec, t = cid.split('_')
        f = features.lecture_features(lec)
        _, floor = oc.detect_onsets(f['total'])
        k = int(round(float(t) / oc.HOP_S))
        items.append({'lec': lec, 'start': q['start'], 'end': q['end'], 'onset': float(t), 'label': lab, 'score': 0.0,
                      'id': cid, 'peak_db': round(float(f['total'][k:k + 10].max() - floor[k]), 1)})
    items.sort(key=lambda c: (c['lec'], c['start']))
    for r, c in enumerate(items, 1):
        c['rank'] = r
    with open(os.path.join(g.REVIEW, 'queue_fa_known.json'), 'w', encoding='utf-8') as f:
        json.dump({'candidates': items}, f, indent=1)
    for c in items:
        vid = ids[c['lec']]
        m, s = divmod(c['onset'], 60)
        print(f"Y{c['rank']:<3} {c['lec']}  {int(m)}:{s:06.3f}  {c['label']:<7} {c['peak_db']:>5.1f} dB  "
              f"https://www.youtube.com/watch?v={vid}&t={max(0, int(c['onset']) - 2)}s")
    print(f'{len(items)} items -> queue_fa_known.json')


if __name__ == '__main__':
    main()
