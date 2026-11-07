"""Local review tool: listen to each candidate, adjust its edges and label it.

Run:  .venv\\Scripts\\python tools\\review_server.py   then open http://127.0.0.1:8765
Decisions go to data/review/decisions.json (and data/labels_review/<lec>.txt for Audacity).
Your own Audacity files in data/labels are only read, never written.
"""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import gulpdata

HOST, PORT = '127.0.0.1', 8765   # localhost only: nothing is reachable from the network
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'review.html')
VALID = gulpdata.GULP_LABELS | gulpdata.KEEP_LABELS | gulpdata.IGNORE_LABELS | {gulpdata.NOTE_LABEL}
MAX_CLIP_S = 30


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

    def do_POST(self):
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
