"""Final check: what the current detector still gets wrong at the default setting (~2 true false
alarms per hour, read-ahead mode with both vetoes), cross-validated so no lecture scores itself.

  missed gulps    labelled gulps it does not mute, grouped by cause:
                    no_onset   the onset detector never fires on it
                    vetoed     a veto (speech 200 ms / breath rule) removed it
                    low_score  the classifier scores it below the threshold
  false alarms    reviewed non-gulp sounds (breath, none, ...) it fires on, after the vetoes

Prints every case and writes 3 of each (different causes first, then the clearest) to
data/review/queue_fa_known.json, shown in the review tool as the final-check list.

Usage: .venv\\Scripts\\python tools\\build_final_check.py
"""
import json
import os

import numpy as np

import gulpdata as g
import onset_classifier as oc
import reanchor as ra
from breath_rule import sound_feats
from train_final import DS_FRAMES, VETO_DB, VETO_FRAMES, shape_veto

import sys
FA = sys.argv[sys.argv.index('--fa') + 1] if '--fa' in sys.argv else '2'


def main():
    model = json.load(open(os.path.join(g.ROOT, 'models', 'detector.json'), encoding='utf-8'))
    eta = model['thresholds_by_fa_per_h'][FA]
    ids = dict(l.split()[:2] for l in open(os.path.join(g.ROOT, 'data', 'playlist_ids.txt'), encoding='utf-8') if l.strip())
    lectures, feats, _ = oc.load_all()
    D = {l: {**ra.lecture_data(feats[l], g.load_labels(l)), 'total': feats[l]['total']} for l in lectures}
    raw = ra.cv(D, lectures, False)
    missed, fas, unl = [], [], []
    for l in lectures:
        d, z = D[l], raw[l]
        tot = d['total']
        on, floor = oc.detect_onsets(tot)
        A = d['A']
        idx = np.clip(A[:, None] + np.arange(VETO_FRAMES)[None, :], 0, len(tot) - 1)
        speech = tot[idx].max(1) - floor[A] > VETO_DB
        di = np.clip(A[:, None] + np.arange(DS_FRAMES)[None, :], 0, len(tot) - 1)
        speech |= (tot[di] <= oc.DIGITAL_SILENCE_DB).any(1)          # end-of-recording cut
        bf = [sound_feats(feats[l], floor, int(k0)) for k0 in on]
        breath = np.array([shape_veto(feats[l], floor, int(k0)) for k0 in on])
        vet = speech | breath[d['grp']]
        labels = g.load_labels(l)
        for gg in ra.merged_gulps(d):                 # parts < 100 ms apart are one gulp
            a = gg['anchors']
            s, e, lab = gg['k0'] * oc.HOP_S, (gg['k1'] + 1) * oc.HOP_S, gg['label']
            if not len(a):
                cause, zmax = 'no_onset', None
            elif (z[a] > eta).any() and not (z[a][~vet[a]] > eta).any():
                cause, zmax = 'vetoed (' + ('breath' if breath[d['grp'][a]].any() else 'speech') + ')', float(z[a].max())
            elif not (z[a] > eta).any():
                cause, zmax = 'low_score', float(z[a].max())
            else:
                continue
            missed.append({'lec': l, 'start': round(s, 3), 'end': round(e, 3), 'label': lab, 'cause': cause,
                           'z': zmax, 'onset': round(on[d['grp'][a[0]]] * oc.HOP_S, 3) if len(a) else None})
        # false alarms: sounds with a reviewed non-gulp label that fire after the vetoes
        bad = np.zeros(d['grp'].max() + 1, bool)
        bad[d['grp'][d['y'] != 0]] = True
        zz = np.where(vet, -np.inf, z)
        best = np.full(len(bad), -np.inf)
        ok = d['y'] == 0
        np.maximum.at(best, d['grp'][ok], zz[ok])
        for gi in np.flatnonzero(~bad & (best > eta)):
            t = on[gi] * oc.HOP_S
            lab = [x for x in labels if x[0] - 0.1 <= t <= x[1] + 0.1 and x[2] not in g.GULP_LABELS]
            x = bf[gi]
            k = on[gi]
            info = {'peak': float(tot[k:k + 40].max() - floor[k]), 'rough': x['lf_rough'], 'tilt': x['tilt'],
                    'lfpk': x['lf_peak'], 'dur': x['dur'], 'gate': x['gate']}
            if lab:
                fas.append({'lec': l, 'start': round(t, 3), 'end': round(t + bf[gi]['dur'] / 1000, 3), 'onset': round(t, 3),
                            'label': lab[0][2], 'cause': 'false alarm', 'z': float(best[gi]), **info})
            else:
                unl.append(float(best[gi]))
    n_g = sum(len(ra.merged_gulps(d)) for d in D.values())
    print(f'default setting ({FA} true FA/h, threshold z {eta:.2f}): {len(missed)} of {n_g} gulps missed; '
          f'{len(fas)} reviewed non-gulp sounds fire\n')
    for m in sorted(missed, key=lambda c: (c['cause'], c['lec'], c['start'])):
        print(f"  missed  {m['lec']} {int(m['start'] // 60)}:{m['start'] % 60:06.3f}  {m['label']:<12} {m['cause']:<18} "
              f"z {m['z'] if m['z'] is None else round(m['z'], 2)}")
    for f in sorted(fas, key=lambda c: -c['z']):
        print(f"  alarm   {f['lec']} {int(f['start'] // 60)}:{f['start'] % 60:06.3f}  {f['label']:<12} z {f['z']:.2f}  "
              f"peak {f['peak']:.0f} dB, lf rough {f['rough']:.1f}, tilt {f['tilt']:.0f}, lf peak {f['lfpk']:.0f}, "
              f"dur {f['dur']} ms, breath gate {'in' if f['gate'] else 'out'}")
    print(f'\n  plus {len(unl)} UNLABELLED sounds above the threshold (audits: about half or more are gulps)')

    # three of each: one per cause first, then the strongest
    # missed: the two most audible gulps the onset detector never sees, and the gulp the
    # classifier likes least; alarms: the three it is most sure of
    sev = {'gulp_high': 0, 'gulp': 1, 'gulp_overlap': 2, 'gulp_low': 3}
    no_on = sorted([m for m in missed if m['cause'] == 'no_onset'], key=lambda c: sev[c['label']])
    low = sorted([m for m in missed if m['cause'] == 'low_score'], key=lambda c: c['z'])
    pick = (no_on[:2] + low)[:3] + sorted(fas, key=lambda c: -c['z'])[:3]
    items = []
    for r, c in enumerate(pick, 1):
        kind = 'K' if c['cause'] != 'false alarm' else 'Y'
        items.append({'lec': c['lec'], 'start': c['start'], 'end': c['end'], 'onset': c['onset'], 'score': 0.0,
                      'label': f"{c['label']} · {c['cause']}", 'id': f"final_{c['lec']}_{c['start']:.3f}", 'rank': r})
        m, s = divmod(c['start'], 60)
        print(f"{kind}{r}: {c['lec']} {int(m)}:{s:06.3f} {c['label']} ({c['cause']})  "
              f"https://www.youtube.com/watch?v={ids[c['lec']]}&t={max(0, int(c['start']) - 2)}s")
    with open(os.path.join(g.REVIEW, 'queue_fa_known.json'), 'w', encoding='utf-8') as f:
        json.dump({'candidates': items}, f, indent=1)


if __name__ == '__main__':
    main()
