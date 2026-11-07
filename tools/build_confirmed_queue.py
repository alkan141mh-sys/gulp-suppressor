"""Review queue of every sound labelled as a gulp (Audacity files and review-tool decisions),
in lecture / time order, with the label's own edges. Writes data/review/queue_confirmed.json.

Usage: .venv\\Scripts\\python tools\\build_confirmed_queue.py
"""
import json
import os

import gulpdata


def main():
    items = []
    for lec in gulpdata.lectures():
        if lec in gulpdata.TEST_LECTURES:
            continue
        for s, e, lab in gulpdata.load_labels(lec):
            if lab in gulpdata.GULP_LABELS:
                items.append({'lec': lec, 'start': round(s, 3), 'end': round(e, 3), 'label': lab,
                              'score': 0.0, 'id': f'conf_{lec}_{s:.3f}'})
    for r, c in enumerate(items, 1):
        c['rank'] = r
    with open(os.path.join(gulpdata.REVIEW, 'queue_confirmed.json'), 'w', encoding='utf-8') as f:
        json.dump({'candidates': items}, f, indent=1)
    print(f'{len(items)} confirmed gulps written to data/review/queue_confirmed.json')


if __name__ == '__main__':
    main()
