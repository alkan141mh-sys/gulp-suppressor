// Read-ahead analysis: pure JavaScript (no browser APIs), tested under Node.
//
// Analyzer   decoded audio arrives as blocks stamped with MEDIA time (the video's own clock,
//            video.currentTime). Contiguous blocks form a segment; each segment has its own
//            Engine (engine.js, the same detector as live mode) run in analysis-only mode, so
//            the mute intervals come out in media time, seconds before they are played.
// Calibrator measures where the audio entering the AudioWorklet sits in media time, by
//            matching its energy envelope to the decoded audio (time-delay estimation).
import { Engine } from './engine.js';

export const FS = 48000;
const BIN = 48;                 // energy bins of 1 ms (48 samples at 48 kHz)
const JOIN_S = 0.010;           // a block within 10 ms of a segment's end continues it
const KEEP_BEHIND_S = 5;        // raw audio kept behind the playhead (for re-analysis)
const FINAL_MARGIN_S = 0.2;     // an ended mute can still be reopened by an anchor 25 ms later
// Mute start in read-ahead mode (tools/eval_mute_variants.py, 153 reviewed gulps): from the
// sound's onset and 30 ms before it, gulps with >=80% of their duration muted go from 60% to
// 90%; audible non-gulp audio muted goes from 0.98 to 1.05 s/h (the 100 ms before an onset are
// quiet by definition, so the pre-roll mostly mutes silence).
const PREROLL_S = 0.030;
const VETO_FRAMES = 20, VETO_DB = 43;

class Segment {
  constructor(t0, engine) {
    this.t0 = t0;               // media time of sample 0
    this.n = 0;                 // samples appended
    this.appends = 0;
    this.engine = engine;
    this.queue = []; this.qo = 0;                     // blocks not yet analysed, offset in queue[0]
    this.queued = 0;
    this.tmp = new Float32Array(128);
    this.chunks = [];           // retained raw audio: { start (sample), data }
    this.bins = new Float32Array(1024); this.nb = 0; this.binBase = 0;   // 1 ms energy sums
    this.acc = 0; this.accN = 0;
    this.mi = 0;                // engine.muteLog entries already converted
    this.mutes = [];            // [start, end, final] in media time
  }
  end() { return this.t0 + this.n / FS; }
  analysedUntil() { return this.t0 + this.engine.sIn / FS; }
}

export class Analyzer {
  constructor(model, fa) {
    this.model = model; this.fa = String(fa);
    this.reset();
  }

  reset() { this.segs = []; this.carry = null; this.duplicates = 0; this.joins = 0; }

  _engine(carry) {
    const e = new Engine(this.model, this.fa, 30, 1, carry);
    e.setRate(1);             // decoded audio is content time: one frame is always 10 ms
    e.fromOnset = true;
    e.preRoll = PREROLL_S;
    // Speech veto (tools/cue_check.py, corrected_roc.py): no labelled gulp (209 measured) passes
    // 40 dB above the floor within 200 ms of its onset; 8 of the 20 audited false alarms do,
    // as the sound turns into loud speech. With the veto, gulps caught at 1 / 2 true false
    // alarms per hour go from 81% / 86% to 85% / 89%, with no labelled gulp lost.
    e.vetoFrames = VETO_FRAMES;
    e.vetoDb = VETO_DB;
    // The user's breath rule (engine.js _breath, tools/breath_rule.py): steady low band and hissy
    // or weak below 3.2 kHz, in unvoiced sounds under 35 dB. Removed 14 of 16 reviewed breaths
    // and no labelled gulp; at 2 true false alarms/h, gulps caught go from 89% to 92%.
    e.breathRule = true;
    // End-of-recording cut (tools/train_final.py): 4 of the 7 reviewed false alarms left at the
    // default setting sit in a video's last 0.2 s, right before digital silence.
    e.dsAhead = 30;
    return e;
  }

  add(time, mono) {
    // time: media time of mono[0]. Returns the segment the block went to (or null: duplicate).
    let seg = this.segs.find(s => Math.abs(s.end() - time) < JOIN_S);
    if (seg) {
      // The first decoded block may be trimmed by the codec delay, so the segment's time origin
      // is taken from the second block, whose timestamp is exact.
      if (seg.appends === 1) seg.t0 = time - seg.n / FS;
      this.joins++;
    } else if (this.segs.some(s => time >= s.t0 - JOIN_S && time < s.end())) {
      this.duplicates++;        // the player re-appended audio we already have
      return null;
    } else {
      const before = this.segs.filter(s => s.end() <= time + JOIN_S).sort((a, b) => b.end() - a.end())[0];
      const carry = before ? before.engine.carryState() : this.carry;
      seg = new Segment(time, this._engine(carry));
      this.segs.push(seg);
      this.segs.sort((a, b) => a.t0 - b.t0);
    }
    seg.appends++;
    this._append(seg, mono);
    return seg;
  }

  // Run the detectors for at most budgetMs; returns true while work remains.
  step(budgetMs = 8, now = () => performance.now()) {
    const t = now();
    for (const seg of this.segs) {
      let k = 0;
      while (seg.queued >= 128) {
        seg.engine.process([this._take(seg)], null);
        if ((++k & 31) === 0) {
          this._collect(seg);
          if (now() - t > budgetMs) return true;
        }
      }
      this._collect(seg);
    }
    return false;
  }

  _take(seg) {
    // Next 128 samples of the segment's queue (a view when they sit in one block).
    const q = seg.queue;
    let out;
    if (q[0].length - seg.qo >= 128) {
      out = q[0].subarray(seg.qo, seg.qo + 128);
      seg.qo += 128;
    } else {
      out = seg.tmp;
      let o = 0;
      while (o < 128) {
        const k = Math.min(128 - o, q[0].length - seg.qo);
        out.set(q[0].subarray(seg.qo, seg.qo + k), o);
        o += k; seg.qo += k;
        if (seg.qo === q[0].length) { q.shift(); seg.qo = 0; }
      }
    }
    if (q.length && seg.qo === q[0].length) { q.shift(); seg.qo = 0; }
    seg.queued -= 128;
    return out;
  }

  _collect(seg) {
    const log = seg.engine.muteLog, cap = this.model.dsp.MUTE_MAX_S;
    const tEng = seg.engine.sIn / FS;
    for (let i = seg.mutes.length; i < log.length; i++) {
      seg.mutes.push(null);
    }
    for (let i = 0; i < log.length; i++) {
      const m = log[i], old = seg.mutes[i];
      if (old && old[2]) continue;
      const end = m.end === null ? m.start + cap : m.end;
      const final = tEng > end + FINAL_MARGIN_S;
      seg.mutes[i] = [seg.t0 + m.start, seg.t0 + end, final];
    }
  }

  // Mute intervals overlapping [a, b] in media time, merged, sorted.
  mutes(a = -Infinity, b = Infinity) {
    const all = [];
    for (const s of this.segs) for (const m of s.mutes) if (m && m[1] > a && m[0] < b) all.push([m[0], m[1]]);
    all.sort((x, y) => x[0] - y[0]);
    const out = [];
    for (const m of all) {
      if (out.length && m[0] <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], m[1]);
      else out.push(m);
    }
    return out;
  }

  segmentAt(t) { return this.segs.find(s => t >= s.t0 && t < s.end()) || null; }

  // How far past t the analysis is complete (decisions need ~0.1 s after an onset), or null.
  coveredUntil(t) {
    const s = this.segmentAt(t);
    if (!s) return null;
    return s.analysedUntil() - 0.15;
  }

  // 1 ms energy bins (sum of squares over 48 samples) on [t0, t0 + n ms), NaN where unknown.
  bins(t0, n) {
    const out = new Float64Array(n).fill(NaN);
    for (const s of this.segs) {
      const i0 = Math.max(0, Math.ceil((s.t0 - t0) * 1000));
      for (let i = i0; i < n; i++) {
        const k = Math.round((t0 + i / 1000 - s.t0) * 1000);
        if (k >= s.nb) break;
        if (k >= s.binBase) out[i] = s.bins[k - s.binBase];
      }
    }
    return out;
  }

  // Forget audio far behind the playhead: raw blocks (re-analysis) and old energy bins.
  evict(playhead) {
    for (const s of this.segs) {
      const keepFrom = Math.floor((playhead - KEEP_BEHIND_S - s.t0) * FS);
      while (s.chunks.length > 1 && s.chunks[0].start + s.chunks[0].data.length < keepFrom) s.chunks.shift();
      const binFrom = Math.floor((playhead - 30 - s.t0) * 1000);
      if (binFrom - s.binBase > 60000) {
        s.bins = s.bins.slice(binFrom - s.binBase); s.binBase = binFrom;
      }
    }
    // Segments that ended long ago and far ahead are both kept: a seek may come back to them.
    this.segs = this.segs.filter(s => s.end() > playhead - 600 || s.t0 > playhead);
  }

  // New operating point: re-run the detector over the audio still held (from a few seconds
  // behind the playhead), each segment with a fresh engine that carries its floor and level.
  setFa(fa) {
    fa = String(fa);
    if (fa === this.fa) return;
    this.fa = fa;
    const old = this.segs;
    this.segs = [];
    for (const s of old) {
      if (!s.chunks.length) continue;
      const carry = s.engine.carryState();
      const t = s.t0 + s.chunks[0].start / FS;
      const ns = new Segment(t, this._engine(carry));
      this.segs.push(ns);
      for (const c of s.chunks) {
        ns.appends = 2;          // the time origin is already exact
        this._append(ns, c.data);
      }
    }
  }

  _append(seg, mono) {
    seg.chunks.push({ start: seg.n, data: mono });
    seg.n += mono.length;
    if (mono.length) { seg.queue.push(mono); seg.queued += mono.length; }
    for (let i = 0; i < mono.length; i++) {
      seg.acc += mono[i] * mono[i];
      if (++seg.accN === BIN) {
        if (seg.nb - seg.binBase >= seg.bins.length) {
          const b = new Float32Array(seg.bins.length * 2); b.set(seg.bins); seg.bins = b;
        }
        seg.bins[seg.nb - seg.binBase] = seg.acc; seg.nb++;
        seg.acc = 0; seg.accN = 0;
      }
    }
  }
}

// ---- clock calibration ----
// The worklet reports, for every 128-sample render quantum, the context time T at its start and
// the mean power of its input. With the anchor (one simultaneous reading of ctx.currentTime and
// video.currentTime, plus the playback rate) each quantum gets an estimated media time
//   m(T) = media0 + (T - ctx0) * rate.
// The estimate is off by an unknown constant (when the element's clock is read, output
// buffering, codec delay). That constant is the lag L maximising the correlation between the
// input's level in dB and the decoded audio's level at m(T) + L.
export class Calibrator {
  constructor() { this.q = []; this.hist = []; this.offset = 0; this.last = null; }

  add(pairs) {
    for (let i = 0; i + 1 < pairs.length; i += 2) this.q.push([pairs[i], pairs[i + 1]]);
    const tEnd = this.q.length ? this.q[this.q.length - 1][0] : 0;
    while (this.q.length && this.q[0][0] < tEnd - 6) this.q.shift();
  }

  newAnchor() { this.hist = []; }

  // Returns { lag, r, n } for the last ~4 s since the anchor, or null when there is too little
  // usable signal (silence, digital silence, no decoded audio there).
  estimate(anchor, analyzer, span = 0.4, sr = FS) {
    if (!anchor || !anchor.playing) return null;
    const rate = anchor.rate;
    const tEnd = this.q.length ? this.q[this.q.length - 1][0] : 0;
    const qs = this.q.filter(([T]) => T >= anchor.ctx + 0.15 && T >= tEnd - 4);
    if (qs.length < 150) return null;              // 0.4 s of audio
    const m = qs.map(([T]) => anchor.media + (T - anchor.ctx) * rate);
    const x = qs.map(([, p]) => 10 * Math.log10(p + 1e-12));
    const lo = Math.min(...m) - span - 0.01, n = Math.ceil((Math.max(...m) - lo + span + 0.02) * 1000);
    const b = analyzer.bins(lo, n);
    const P = new Float64Array(n + 1), V = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) {
      const ok = Number.isFinite(b[i]);
      P[i + 1] = P[i] + (ok ? b[i] : 0); V[i + 1] = V[i] + (ok ? 1 : 0);
    }
    const w = Math.max(1, Math.round(128 / sr * rate * 1000));
    const use = x.map(v => v > -75);
    let best = null;
    const nl = Math.round(span * 1000);
    for (let L = -nl; L <= nl; L++) {
      let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, k = 0;
      for (let i = 0; i < qs.length; i++) {
        if (!use[i]) continue;
        const j = Math.round((m[i] - lo) * 1000) + L;
        if (j < 0 || j + w > n || V[j + w] - V[j] < w) continue;
        const y = 10 * Math.log10((P[j + w] - P[j]) / (w * 48) + 1e-12);
        sx += x[i]; sy += y; sxx += x[i] * x[i]; syy += y * y; sxy += x[i] * y; k++;
      }
      if (k < 120) continue;
      const vx = sxx - sx * sx / k, vy = syy - sy * sy / k;
      if (vx <= 0 || vy <= 0) continue;
      const r = (sxy - sx * sy / k) / Math.sqrt(vx * vy);
      if (!best || r > best.r) best = { lag: L / 1000, r, n: k, sdx: Math.sqrt(vx / k) };
    }
    this.last = best;
    return best;
  }

  // Accept a good estimate; the offset is the median of the last three accepted ones.
  update(est, minR = 0.8) {
    if (!est || est.r < minR || est.sdx < 3) return false;
    this.hist.push(est.lag);
    if (this.hist.length > 3) this.hist.shift();
    const s = [...this.hist].sort((a, b) => a - b);
    this.offset = s[Math.floor(s.length / 2)];
    return true;
  }
}
