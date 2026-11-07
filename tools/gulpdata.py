"""Shared access to lecture audio, label files and review decisions."""
import datetime
import glob
import io
import json
import os
import re
import threading
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
LABELS = os.path.join(ROOT, 'data', 'labels')          # Audacity exports (yours)
REVIEW = os.path.join(ROOT, 'data', 'review')          # review tool state
REVIEW_LABELS = os.path.join(ROOT, 'data', 'labels_review')  # same decisions, Audacity format
CACHE = os.path.join(ROOT, 'data', 'cache')

# Held-out test set. Nothing may train on these or suggest candidates in them,
# otherwise the final miss / false-alarm figures would be optimistic.
TEST_LECTURES = {'037', '038', '039', '040'}

GULP_LABELS = {'gulp_low', 'gulp', 'gulp_high', 'gulp_overlap'}   # H1
KEEP_LABELS = {'not_gulp', 'breath', 'mouth', 'none'}             # H0, reviewed
# 'gulp_click': the click that often follows a gulp. Neither H1 nor H0 until it is decided
# whether clicks should be muted too; as an ignored label it teaches the model nothing yet.
CLICK_LABEL = 'gulp_click'
IGNORE_LABELS = {'unsure', CLICK_LABEL}                           # neither
# A 'note' decision is a comment on an existing label, not a label: its tags say why a gulp
# was missed or what a multi-burst label holds. Two tags correct the gulp label the note is
# about, without editing your Audacity files: 'not_gulp' turns it into not_gulp, and
# 'edges_wrong' replaces its start and end with the span drawn when the note was saved.
NOTE_LABEL = 'note'
NOTE_TAGS = {'after_speech', 'quiet', 'multi_burst', 'unusual', 'not_gulp', 'edges_wrong',
             'one_gulp', 'two_sounds', 'checked_ok'}


def lectures():
    """Lecture ids ('001'...) that have a WAV in data/raw."""
    ids = []
    for p in glob.glob(os.path.join(RAW, '*.wav')):
        m = re.match(r'(\d{3}) ', os.path.basename(p))
        if m:
            ids.append(m.group(1))
    return sorted(ids)


def wav_path(lec):
    hits = glob.glob(os.path.join(RAW, f'{lec} *.wav'))
    if len(hits) != 1:
        raise FileNotFoundError(f'expected one WAV for lecture {lec}, found {len(hits)}')
    return hits[0]


def read_wav(lec, t0=None, t1=None):
    """Return (int16 samples, fs) for the whole lecture or the span [t0, t1) in seconds."""
    with wave.open(wav_path(lec), 'rb') as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, 'expected mono 16-bit'
        fs, n = w.getframerate(), w.getnframes()
        a = 0 if t0 is None else min(n, max(0, int(round(t0 * fs))))
        b = n if t1 is None else min(n, max(a, int(round(t1 * fs))))
        w.setpos(a)
        x = np.frombuffer(w.readframes(b - a), dtype='<i2')
    return x, fs


def wav_bytes(x, fs):
    """Encode int16 samples as a WAV file in memory (for the browser)."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(x.astype('<i2').tobytes())
    return buf.getvalue()


def normalise_label(s):
    return s.strip().lower().replace(' ', '_')


def read_audacity_labels(path):
    """Parse an Audacity label export: 'start<TAB>end<TAB>label' per line.

    Lines starting with a backslash hold spectral-selection frequencies and are skipped.
    """
    out = []
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line.strip() or line.startswith('\\'):
                continue
            s, e, lab = line.split('\t')
            out.append((float(s), float(e), normalise_label(lab)))
    return out


def spans_of(d):
    """A decision's intervals: several when the review tool's multi-span selection was used."""
    return [tuple(s) for s in d.get('spans', [])] or [(d['start'], d['end'])]


def audit_controls():
    """Ids of the audit's gulp controls (already-labelled gulps shown again, blind)."""
    p = os.path.join(REVIEW, 'audit_key.json')
    if not os.path.exists(p):
        return set()
    with open(p, encoding='utf-8') as f:
        return {cid for cid, kind in json.load(f)['key'].items() if kind == 'gulp_control'}


def load_labels(lec, exclude=None):
    """All labels for a lecture: your Audacity file plus review decisions, de-duplicated.

    A decision with several spans gives one label per span.

    exclude: optional predicate on a decision id; matching decisions are left out (used to
    reproduce a model exactly as it was before some decisions existed).
    """
    items = []
    p = os.path.join(LABELS, f'{lec}.txt')
    if os.path.exists(p):
        items += read_audacity_labels(p)
    notes = []
    # The blind audit (build_audit.py) re-presents some labelled gulps as controls; their audit
    # answers would duplicate those gulps with other edges, so they are not labels.
    controls = audit_controls()
    for cid, d in load_decisions().items():
        if d['lec'] != lec or (exclude and exclude(cid)) or cid in controls:
            continue
        if d['label'] == NOTE_LABEL:
            notes.append((cid, d))
            continue
        items += [(s, e, d['label']) for s, e in spans_of(d)]
    extra = []
    for cid, d in notes:
        tags = d.get('tags', [])
        # Which gulp label is the note about? Queue ids end with that label's original start
        # (conf_005_284.614, miss_..., multi_...); otherwise fall back to overlap.
        try:
            orig = float(cid.rsplit('_', 1)[1])
        except (IndexError, ValueError):
            orig = None
        for i, (s, e, lab) in enumerate(items):
            if lab not in GULP_LABELS:
                continue
            about = abs(s - orig) < 0.01 if orig is not None else (s < d['end'] and d['start'] < e)
            if not about:
                continue
            if 'not_gulp' in tags:
                items[i] = (s, e, 'not_gulp')
            elif 'edges_wrong' in tags:
                sp = spans_of(d)                     # redrawn edges; several spans split the label
                items[i] = (sp[0][0], sp[0][1], lab)
                extra += [(a, b, lab) for a, b in sp[1:]]
    items += extra
    items.sort()
    # The same label can arrive twice if a review file is re-exported from Audacity.
    out = []
    for it in items:
        if out and it[2] == out[-1][2] and abs(it[0] - out[-1][0]) < 0.01 and abs(it[1] - out[-1][1]) < 0.01:
            continue
        out.append(it)
    return out


_lock = threading.Lock()
DECISIONS = os.path.join(REVIEW, 'decisions.json')


def load_decisions():
    if not os.path.exists(DECISIONS):
        return {}
    with open(DECISIONS, encoding='utf-8') as f:
        return json.load(f)


def set_decision(cid, lec, start, end, label, recheck=False, tags=None, spans=None):
    """Store (or clear, when label is None) one review decision.

    A changed or re-checked decision keeps its earlier versions in 'history', so labelling
    consistency can be measured later (tools/recheck_agreement.py).
    decisions.json is the source of truth; data/labels_review/<lec>.txt is rewritten from it
    so the reviewed labels can also be imported into Audacity.
    """
    with _lock:
        d = load_decisions()
        if label is None:
            d.pop(cid, None)
        else:
            old = d.get(cid)
            new = {'lec': lec, 'start': round(start, 6), 'end': round(end, 6), 'label': label,
                   'time': datetime.datetime.now().isoformat(timespec='seconds')}
            if tags:
                new['tags'] = sorted(tags)
            if spans and len(spans) > 1:
                new['spans'] = sorted([round(a, 6), round(b, 6)] for a, b in spans)
                new['start'], new['end'] = new['spans'][0][0], max(b for _, b in new['spans'])
            if old:
                history = old.pop('history', [])
                key = lambda v: (v['label'], v['start'], v['end'], v.get('spans'))
                if recheck or key(old) != key(new):
                    history.append(old)
                if history:
                    new['history'] = history
                if recheck or old.get('rechecked'):
                    new['rechecked'] = True
            d[cid] = new
        os.makedirs(REVIEW, exist_ok=True)
        tmp = DECISIONS + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=1, sort_keys=True)
        os.replace(tmp, DECISIONS)   # atomic: a crash never leaves half a file

        os.makedirs(REVIEW_LABELS, exist_ok=True)
        rows = sorted((s, e, v['label']) for v in d.values()
                      if v['lec'] == lec and v['label'] != NOTE_LABEL     # notes are not labels
                      for s, e in spans_of(v))
        with open(os.path.join(REVIEW_LABELS, f'{lec}.txt'), 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(f'{s:.6f}\t{e:.6f}\t{lab}\n' for s, e, lab in rows)
        return d
