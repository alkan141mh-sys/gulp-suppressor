"""Local review tool: listen to each candidate, adjust its edges and label it.

Run:  .venv\\Scripts\\python tools\\review_server.py   then open http://127.0.0.1:8765
Decisions go to data/review/decisions.json (and data/labels_review/<lec>.txt for Audacity).
Your own Audacity files in data/labels are only read, never written.

Held-out test labelling: http://127.0.0.1:8765/label  (label_test.html). The four test lectures
on one timeline, labelled by ear and eye only: nothing from any detector is shown. Labels go to
data/test_labels/labels.json (and <lec>.txt in Audacity format); nothing else reads them.
"""
import json
import os
import re
import tempfile
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import gulpdata

HOST, PORT = '127.0.0.1', 8765   # localhost only: nothing is reachable from the network
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'review.html')
VALID = gulpdata.GULP_LABELS | gulpdata.KEEP_LABELS | gulpdata.IGNORE_LABELS | {gulpdata.NOTE_LABEL}
MAX_CLIP_S = 30
TEST_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'label_test.html')
TEST_DIR = os.path.join(gulpdata.ROOT, 'data', 'test_labels')
TEST_FILE = os.path.join(TEST_DIR, 'labels.json')
TEST_ORDER = sorted(gulpdata.TEST_LECTURES)
TEST_VALID = {'gulp_low', 'gulp', 'gulp_high', 'gulp_overlap', 'unsure'}
PEAK_MS = 5                      # one waveform column per 5 ms
PEAK_SCALE = 2.5                 # byte = 2.5 * 20 log10(peak sample), 0..226


def wav_info(lec):
    with wave.open(gulpdata.wav_path(lec), 'rb') as w:
        return w.getframerate(), w.getnframes()


def test_peaks(lec):
    """Peak level per 5 ms as one byte (dB above 1 LSB, times 2.5), cached."""
    p = os.path.join(gulpdata.CACHE, f'test_peaks_{lec}.bin')
    if not os.path.exists(p):
        x, fs = gulpdata.read_wav(lec)
        hop = fs * PEAK_MS // 1000
        n = len(x) // hop
        pk = np.abs(x[:n * hop].astype(np.int32)).reshape(n, hop).max(1)
        b = np.round(PEAK_SCALE * 20 * np.log10(np.maximum(pk, 1))).astype(np.uint8)
        os.makedirs(gulpdata.CACHE, exist_ok=True)
        b.tofile(p)
    with open(p, 'rb') as f:
        return f.read()


def test_state():
    if os.path.exists(TEST_FILE):
        with open(TEST_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {'labels': {l: [] for l in TEST_ORDER}, 'listened': {l: [] for l in TEST_ORDER}}


def save_test_state(st):
    """Validate, then write labels.json atomically plus one Audacity file per lecture."""
    out = {'labels': {}, 'listened': {}}
    for l in TEST_ORDER:
        labs = st['labels'].get(l, [])
        if not isinstance(labs, list) or len(labs) > 2000:
            raise ValueError
        clean = []
        for a, b, lab in labs:
            a, b = float(a), float(b)
            if lab not in TEST_VALID or not 0 <= a < b or b - a > 10:
                raise ValueError
            clean.append([round(a, 3), round(b, 3), lab])
        out['labels'][l] = sorted(clean)
        heard = st['listened'].get(l, [])
        if not isinstance(heard, list) or len(heard) > 5000:
            raise ValueError
        out['listened'][l] = [[round(float(a), 2), round(float(b), 2)] for a, b in heard if 0 <= float(a) <= float(b)]
    os.makedirs(TEST_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=TEST_DIR, suffix='.tmp')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, TEST_FILE)
    for l in TEST_ORDER:
        with open(os.path.join(TEST_DIR, f'{l}.txt'), 'w', encoding='utf-8') as f:
            for a, b, lab in out['labels'][l]:
                f.write(f'{a:.6f}\t{b:.6f}\t{lab}\n')
    return out


SOURCES = [('candidates.json', 'ranker'),            # find_candidates.py
           ('candidates_fa.json', 'false_alarm'),    # export_false_alarms.py
           ('candidates_onset.json', 'onset_fa'),    # export_onset_false_alarms.py
           ('queue_hidden.json', 'hidden'),          # build_queues.py: likely unlabelled gulps
           ('queue_missed.json', 'missed'),          # build_queues.py: labelled gulps the detector misses
           ('queue_multi.json', 'multi'),            # build_queues.py: multi-burst gulp labels
           ('queue_confirmed.json', 'confirmed'),    # build_confirmed_queue.py: every gulp label
           ('queue_audit.json', 'audit'),            # build_audit.py: blind audit of 5 FA/h alarms
           ('queue_fa_known.json', 'fa_known')]      # build_fa_known.py: audit alarms that were not gulps


def candidates():
    out = []
    for name, source in SOURCES:
        p = os.path.join(gulpdata.REVIEW, name)
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                out += [{**c, 'source': source} for c in json.load(f)['candidates']]
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), 'application/json')

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == '/':
            with open(PAGE, 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')
        elif u.path == '/label':
            with open(TEST_PAGE, 'rb') as f:
                self._send(200, f.read(), 'text/html; charset=utf-8')
        elif u.path == '/api/test/info':
            info = []
            for l in TEST_ORDER:
                fs, n = wav_info(l)
                info.append({'lec': l, 'dur': n / fs})
            self._json({'lectures': info, 'peak_ms': PEAK_MS, 'peak_scale': PEAK_SCALE, **test_state()})
        elif u.path == '/api/test/peaks':
            lec = q.get('lec', [''])[0]
            if lec not in TEST_ORDER:
                return self._json({'error': 'bad request'}, 400)
            self._send(200, test_peaks(lec), 'application/octet-stream')
        elif u.path == '/api/test/audio':
            lec = q.get('lec', [''])[0]
            if lec not in TEST_ORDER:
                return self._json({'error': 'bad request'}, 400)
            self._file_range(gulpdata.wav_path(lec), 'audio/wav')
        elif u.path == '/api/state':
            self._json({'candidates': candidates(), 'decisions': gulpdata.load_decisions(),
                        'labels': sorted(VALID)})
        elif u.path == '/api/audio':
            try:
                lec = q['lec'][0]
                t0, t1 = float(q['t0'][0]), float(q['t1'][0])
                if not re.fullmatch(r'\d{3}', lec) or not 0 <= t0 < t1 or t1 - t0 > MAX_CLIP_S:
                    raise ValueError
                x, fs = gulpdata.read_wav(lec, t0, t1)
            except (KeyError, ValueError, FileNotFoundError):
                return self._json({'error': 'bad request'}, 400)
            self._send(200, gulpdata.wav_bytes(x, fs), 'audio/wav')
        else:
            self._json({'error': 'not found'}, 404)

    def _file_range(self, path, ctype):
        """Serve a file with HTTP Range support, so the browser can seek in a long lecture."""
        size = os.path.getsize(path)
        a, b, code = 0, size - 1, 200
        m = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range', '') or '')
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                a = int(m.group(1))
                b = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                a = max(0, size - int(m.group(2)))
            if a > b:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(b - a + 1))
        if code == 206:
            self.send_header('Content-Range', f'bytes {a}-{b}/{size}')
        self.end_headers()
        try:
            with open(path, 'rb') as f:
                f.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass                                   # the browser cancelled the request (a seek)

    def do_POST(self):
        if urlparse(self.path).path == '/api/test/save':
            try:
                st = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                out = save_test_state(st)
            except (KeyError, TypeError, ValueError):
                return self._json({'error': 'bad request'}, 400)
            return self._json({'ok': True, 'n': sum(len(v) for v in out['labels'].values())})
        if urlparse(self.path).path != '/api/decide':
            return self._json({'error': 'not found'}, 404)
        try:
            d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            label = d['label']
            if label is not None and label not in VALID:
                raise ValueError
            start, end = float(d['start']), float(d['end'])
            if not re.fullmatch(r'\d{3}', d['lec']) or not 0 <= start < end:
                raise ValueError
            if d['lec'] in gulpdata.TEST_LECTURES:
                raise ValueError
            tags = d.get('tags') or []
            if not isinstance(tags, list) or not set(tags) <= gulpdata.NOTE_TAGS:
                raise ValueError
            if label == gulpdata.NOTE_LABEL and not tags:
                raise ValueError                      # an empty note says nothing
            spans = d.get('spans') or []
            if not isinstance(spans, list) or len(spans) > 8:
                raise ValueError
            spans = [(float(a), float(b)) for a, b in spans]
            if any(not 0 <= a < b for a, b in spans):
                raise ValueError
            decisions = gulpdata.set_decision(str(d['id']), d['lec'], start, end, label,
                                              recheck=bool(d.get('recheck')), tags=tags, spans=spans)
        except (KeyError, TypeError, ValueError):
            return self._json({'error': 'bad request'}, 400)
        self._json({'decisions': decisions})

    def log_request(self, code='-', size='-'):   # keep the console quiet apart from errors
        if str(code)[:1] in '45':
            super().log_request(code, size)


if __name__ == '__main__':
    print(f'Review tool: http://{HOST}:{PORT}  (Ctrl+C to stop)')
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
