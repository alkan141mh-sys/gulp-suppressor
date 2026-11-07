// Gulp suppressor engine, a line-by-line port of tools/live.py (Engine and Resampler).
// Pure JavaScript, no browser APIs: it runs inside the AudioWorklet and under Node for the
// equivalence test (extension/test/run_lecture.mjs). All constants come from model.json,
// exported by tools/export_extension.py, so both engines use the same numbers.

class FFT {
  // In-place iterative radix-2 complex FFT of fixed size n.
  constructor(n) {
    this.n = n;
    this.rev = new Uint32Array(n);
    for (let i = 0, j = 0; i < n; i++) {
      this.rev[i] = j;
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) {
      this.cos[k] = Math.cos(-2 * Math.PI * k / n);
      this.sin[k] = Math.sin(-2 * Math.PI * k / n);
    }
  }

  transform(re, im) {
    const n = this.n;
    for (let i = 0; i < n; i++) {
      const j = this.rev[i];
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step], wi = this.sin[k * step];
          const p = i + k, q = p + half;
          const tr = re[q] * wr - im[q] * wi, ti = re[q] * wi + im[q] * wr;
          re[q] = re[p] - tr; im[q] = im[p] - ti;
          re[p] += tr; im[p] += ti;
        }
      }
    }
  }
}

class Resampler {
  // Streaming 48 -> 16 kHz that reproduces scipy's resample_poly(x, 1, 3).
  // y[n] = sum_j H[3n + HALF - j] x[j], x = 0 before the start.
  constructor(d) {
    this.h = Float64Array.from(d.taps).reverse();   // dot product with x[3n-HALF .. 3n+HALF]
    this.half = d.HALF;
    this.down = d.DOWN;
    this.buf = new Float64Array(d.HALF);            // zeros stand for x[-HALF .. -1]
    this.start = -d.HALF;
    this.n = 0;
  }

  push(x) {
    const nb = new Float64Array(this.buf.length + x.length);
    nb.set(this.buf); nb.set(x, this.buf.length);
    this.buf = nb;
    const last = this.start + this.buf.length - 1;
    const nEnd = Math.floor((last - this.half) / this.down);
    const out = [];
    const L = this.h.length;
    for (let n = this.n; n <= nEnd; n++) {
      const lo = this.down * n - this.half - this.start;
      let s = 0;
      for (let j = 0; j < L; j++) s += this.buf[lo + j] * this.h[j];
      out.push(s);
    }
    if (nEnd >= this.n) {
      this.n = nEnd + 1;
      const keep = this.down * this.n - this.half - this.start;
      this.buf = this.buf.slice(keep);
      this.start += keep;
    }
    return out;
  }
}

export class Engine {
  // carry: { floor, p95, rate } from a previous engine. After a seek, pause or play the stream
  // restarts, but the room floor and speech level are those of the same video, so the new engine
  // can start detecting after WARM_CARRY frames instead of 2 s.
  constructor(model, faPerH = '15', attenDb = 30, channels = 2, carry = null) {
    const d = model.dsp, p = model.params;
    this.d = d; this.p = p;
    this.mu = Float64Array.from(model.mu);
    this.sd = Float64Array.from(model.sd);
    this.beta = Float64Array.from(model.beta);
    this.setOperatingPoint(model, faPerH);
    this.setAttenuation(attenDb);
    this.p95 = carry ? carry.p95 : model.p95_default;
    this.carry = carry;
    this.warm = carry ? 15 : 2 * p.FLOOR_N;    // frames before onsets may fire
    // Playback speed. Frames advance by HOP / rate samples, so one frame always covers 10 ms of
    // CONTENT: at 1.5x, 10 ms of content plays in 6.7 ms, and YouTube compresses time with the
    // pitch kept. Without this the 100 ms patch would hold 150 ms of content.
    this.rate = carry ? carry.rate : 1;
    this.pos = 0;                              // 16 kHz position of the next frame start
    this.lastFloor = carry ? carry.floor : 0;
    this.muteLog = [];
    this.delay = Math.round(d.DELAY_MS / 1000 * d.FS_IN);
    this.step = 1 / (d.RAMP_MS / 1000 * d.FS_IN);
    this.rs = new Resampler(d);
    this.fft = new FFT(d.NFFT);
    this.re = new Float64Array(d.NFFT);
    this.im = new Float64Array(d.NFFT);
    this.win = Float64Array.from(d.window);
    this.edges = Int32Array.from(d.band_edges);
    this.nb = this.edges.length - 1;
    this.y16 = [];
    this.yOff = 0;
    this.k = 0;
    this.R = 8192;
    this.bands = new Float64Array(this.R * this.nb);
    this.flux = new Float64Array(this.R);
    this.total = new Float64Array(this.R);
    this.harm = new Float64Array(this.R);
    this.sm = new Float64Array(this.R);
    this.floor = new Float64Array(this.R);
    this.tstart = new Float64Array(this.R);   // input (wall) time of each frame's start, s
    this.above = new Uint8Array(this.R);
    this.dig = new Uint8Array(this.R);
    this.sounds = []; this.pending = []; this.mutes = [];
    this.channels = channels;
    this.fifoLen = this.delay + 1024;
    this.fifo = Array.from({ length: channels }, () => new Float32Array(this.fifoLen));
    this.fifoW = this.delay;        // write position; reads start at 0 => delay samples of silence
    this.fifoR = 0;
    this.sIn = 0;
    this.sInBlock = 0;
    this.gain = 1;
    this.bypass = false;
    // Read-ahead only (ahead_core.js sets these): a mute may start before the decision that
    // causes it. fromOnset: a firing secondary anchor mutes from its sound's onset, not from
    // itself; preRoll: seconds the mute starts before that.
    this.fromOnset = false;
    this.preRoll = 0;
    this.vetoDb = d.VETO_DB ?? 45;
    // Frames the veto looks at. Live mode: the decision window itself (n_after). Read-ahead
    // mode looks 200 ms ahead (ahead_core.js), waiting for those frames before deciding.
    this.vetoFrames = p.n_after;
    this.breathRule = false;        // read-ahead only: needs the sound's first 400 ms
    // Read-ahead only: digital silence this many frames after the anchor means the recording
    // ends (or is cut) right there; that cut is not a gulp (no labelled gulp has any within 1 s).
    this.dsAhead = 0;
    this.events = [];               // [input time s, kind, z, fired]
    this.late = 0;
    this.nMutes = 0;
  }

  setOperatingPoint(model, faPerH) {
    // An unknown setting (e.g. one saved by an older version) falls back to the model's default.
    const th = model.thresholds_by_fa_per_h;
    this.eta = th[String(faPerH)] ?? th[model.default_fa ?? Object.keys(th)[0]];
  }
  setAttenuation(db) { this.gMute = Math.pow(10, -db / 20); }
  setRate(r) { this.rate = Math.min(4, Math.max(0.25, r || 1)); }
  carryState() { return { floor: this.lastFloor, p95: this.p95, rate: this.rate }; }

  _stats(fr) {
    // features.frame_stats for one frame: bands, total, harmonicity.
    const d = this.d, N = d.NFFT, re = this.re, im = this.im;
    re.fill(0); im.fill(0);
    for (let i = 0; i < d.WIN; i++) re[i] = fr[i] * this.win[i];
    this.fft.transform(re, im);
    const half = N / 2 + 1;
    const P = new Float64Array(half);
    for (let k = 0; k < half; k++) P[k] = re[k] * re[k] + im[k] * im[k];
    // Each band summed directly (see features.frame_stats: a cumulative-sum difference loses
    // precision in the weak bands).
    const bands = new Float64Array(this.nb);
    for (let b = 0; b < this.nb; b++) {
      let s = 0;
      for (let k = this.edges[b]; k < this.edges[b + 1]; k++) s += P[k];
      bands[b] = 10 * Math.log10(s + 1e-12);
    }
    let pin = 0;
    for (let k = this.edges[0]; k < this.edges[this.nb]; k++) pin += P[k];
    const total = 10 * Math.log10(pin + 1e-12);
    // Autocorrelation = inverse FFT of the power spectrum (real and symmetric, so a forward
    // FFT of the full symmetric spectrum divided by N gives the same).
    re.fill(0); im.fill(0);
    for (let k = 0; k < half; k++) re[k] = P[k];
    for (let k = 1; k < N / 2; k++) re[N - k] = P[k];
    this.fft.transform(re, im);
    const wac = d.win_ac, ac0 = re[0] / N / wac[0] + 1e-12;
    let best = -Infinity;
    for (let lag = d.lag_min; lag <= d.lag_max; lag++) {
      const r = re[lag] / N / wac[lag] / ac0;
      if (r > best) best = r;
    }
    return { bands, total, harm: Math.min(1, Math.max(0, best)) };
  }

  _frame(fr) {
    const k = this.k, p = this.p, R = this.R, nb = this.nb;
    const { bands: b, total: t, harm: h } = this._stats(fr);
    const i = k % R;
    this.bands.set(b, i * nb);
    this.total[i] = t; this.harm[i] = h;
    if (k) {
      const pi = ((k - 1) % R) * nb;
      let f = 0;
      for (let j = 0; j < nb; j++) f += Math.max(b[j] - this.bands[pi + j], 0);
      this.flux[i] = f;
    } else this.flux[i] = 0;
    const n5 = Math.min(5, k + 1);
    let s = 0;
    for (let j = 0; j < n5; j++) s += this.total[(k - j) % R];
    this.sm[i] = s / n5;
    const flN = p.FLOOR_N;
    if (this.carry && k < 10) this.floor[i] = this.carry.floor;   // tracker still empty after a reset
    else if (k === 0) this.floor[i] = this.sm[i];
    else {
      let m = Infinity;
      for (let j = 1; j <= Math.min(flN, k); j++) m = Math.min(m, this.sm[(k - j) % R]);
      this.floor[i] = m;
    }
    this.lastFloor = this.floor[i];
    this.above[i] = t >= this.floor[i] + p.ONSET_DB ? 1 : 0;
    this.dig[i] = t <= p.DIGITAL_SILENCE_DB ? 1 : 0;
    const q = Math.round(p.QUIET_MS / 10);
    let quietBefore = true;
    for (let j = 1; j <= Math.min(q, k); j++) if (this.above[(k - j) % R]) { quietBefore = false; break; }
    let enabled = k >= this.warm;
    if (enabled) for (let j = 0; j <= Math.min(2 * flN, k); j++) if (this.dig[(k - j) % R]) { enabled = false; break; }
    if (k % 100 === 0 && k >= this.d.P95_FRAMES) {
      const last = new Float64Array(this.d.P95_FRAMES);
      for (let j = 0; j < last.length; j++) last[j] = this.total[(k - j) % R];
      this.p95 = percentile(last, 95);
    }

    for (const snd of [...this.sounds]) {
      if (k > snd.k0 + p.MAX_AFTER) { this._drop(snd); continue; }
      if (t < snd.floor + p.ONSET_DB) {
        snd.quiet += 1;
        if (snd.quiet >= p.END_QUIET) { this._drop(snd); continue; }
      } else snd.quiet = 0;
      const weak = p.LEADIN_DB === null || snd.top < snd.floor + p.LEADIN_DB;
      if (weak && t >= snd.top + p.JUMP_DB && k - snd.last >= p.MIN_GAP && snd.n2 < p.MAX_SECONDARY) {
        this.pending.push({ k, kind: 1, k0: snd.k0 });
        snd.last = k; snd.n2 += 1;
      }
      snd.top = Math.max(snd.top, t);
    }
    if (this.above[i] && quietBefore && enabled) {
      this.sounds.push({ k0: k, floor: this.floor[i], top: t, last: k, quiet: 0, n2: 0 });
      this.pending.push({ k, kind: 0, k0: k });
    }

    const wait = Math.max(p.n_after, this.vetoFrames, this.breathRule ? BREATH_FRAMES : 0, this.dsAhead) - 1;
    const ready = this.pending.filter(a => a.k + wait <= k);
    this.pending = this.pending.filter(a => a.k + wait > k);
    for (const a of ready) this._decide(a);
    for (const m of this.mutes) {
      if (m.end === null && m.seen < k) { m.seen = k; this._muteStep(m, k); }
    }
    this.k += 1;
  }

  _muteStep(m, j) {
    // Same rules as live.py's _mute_step: back near the floor, speech resuming, or the cap.
    // Durations are content time, so wall-clock limits shrink with the playback rate.
    const p = this.p, d = this.d, R = this.R;
    const t = this.total[j % R], h = this.harm[j % R];
    if (t < m.floor + p.ONSET_DB) {
      m.quiet += 1;
      if (m.quiet >= d.MUTE_END_QUIET) m.end = this.tstart[(j - d.MUTE_END_QUIET + 1) % R] + 0.025;
    } else m.quiet = 0;
    if (j > m.k && h >= d.SPEECH_HARM && t >= m.floor + d.SPEECH_DB) {
      m.speech += 1;
      if (m.speech >= d.SPEECH_FRAMES && m.end === null) m.end = Math.max(m.start, this.tstart[(j - d.SPEECH_FRAMES) % R]);
    } else m.speech = 0;
    if (m.end === null && (this.tstart[j % R] + 0.025) - m.start >= d.MUTE_MAX_S / this.rate) {
      m.end = m.start + d.MUTE_MAX_S / this.rate;
    }
  }

  _drop(snd) { this.sounds.splice(this.sounds.indexOf(snd), 1); }

  // The user's breath rule (tools/breath_rule.py, same numbers): a sound that is unvoiced and
  // below 35 dB, whose low band (< 3.2 kHz) is steady (mean |change| < 2 dB per 10 ms) and
  // which is either hissy (energy above 3.2 kHz within 18 dB of the energy below) or weak below
  // 3.2 kHz (< 16 dB above the floor), is a breath, not a gulp. Measured from the sound's onset
  // k0 until 100 ms back near the floor, at most 400 ms.
  // Second read-ahead rule (tools/click_before_speech.py): loud speech (>= 43 dB) arrives
  // within 400 ms, but the sound that fired ended before it and is weak below 3.2 kHz
  // (< 14 dB): a small click or lip sound before a word. Removed 12 of 185 reviewed non-gulp
  // sounds and none of 208 gulps.
  _breath(k0) {
    const s = this._sound(k0);
    if (s.speechAt !== null && s.dur < s.speechAt && s.lfPeak < 14) return true;
    if (!s.gate) return false;
    return s.rough < 2.0 && (s.tilt > -18 || s.lfPeak < 16);
  }

  // Shape of the sound starting at k0 (breath_rule.sound_feats, same numbers).
  _sound(k0) {
    const R = this.R, nb = this.nb, fl = this.floor[k0 % R], p = this.p;
    const last = Math.min(k0 + BREATH_FRAMES, this.k + 1);     // frames k0 .. last-1 exist
    let quiet = 0, end = last, maxLev = -Infinity, voiced = 0, speechAt = null;
    for (let k = k0 + 1; k < last; k++) {
      if (this.total[k % R] < fl + p.ONSET_DB) {
        if (++quiet >= 10) { end = k - 9; break; }
      } else quiet = 0;
    }
    for (let k = k0; k < last; k++) {
      const lev = this.total[k % R] - fl;
      maxLev = Math.max(maxLev, lev);
      if (this.harm[k % R] >= 0.8 && lev >= 25) voiced++;
      if (speechAt === null && lev >= 43) speechAt = (k - k0) * 10;
    }
    end = Math.max(end, k0 + 2);
    let sLf = 0, sHf = 0, prev = null, dsum = 0, n = 0, lfMax = -Infinity;
    for (let k = k0; k < end; k++) {
      let lf = 0, hf = 0;
      for (let b = 0; b < nb; b++) {
        const v = Math.pow(10, this.bands[(k % R) * nb + b] / 10);
        if (b < BREATH_LF_BANDS) lf += v; else hf += v;
      }
      sLf += lf; sHf += hf;
      const lfDb = 10 * Math.log10(lf + 1e-12);
      lfMax = Math.max(lfMax, lfDb);
      if (prev !== null) { dsum += Math.abs(lfDb - prev); n++; }
      prev = lfDb;
    }
    return { gate: voiced === 0 && maxLev < 35, rough: n ? dsum / n : 0,
             tilt: 10 * Math.log10(sHf + 1e-12) - 10 * Math.log10(sLf + 1e-12),
             lfPeak: lfMax - fl, dur: (end - k0) * 10, speechAt };
  }

  _decide(a) {
    const p = this.p, R = this.R, nb = this.nb, ka = a.k;
    const fl = this.floor[ka % R];
    const x = new Float64Array(this.mu.length);
    let o = 0;
    for (let j = -p.PRE; j < p.n_after; j++) {
      const fi = (Math.min(Math.max(ka + j, 0), this.k) % R);
      for (let b = 0; b < nb; b++) x[o++] = this.bands[fi * nb + b] - fl;
      x[o++] = this.harm[fi];
      x[o++] = this.flux[fi];
    }
    x[o++] = fl - this.p95;
    x[o++] = a.kind;
    x[o++] = (ka - a.k0) * 10;
    let z = this.beta[this.beta.length - 1];
    for (let j = 0; j < x.length; j++) {
      const v = Math.min(4, Math.max(-4, (x[j] - this.mu[j]) / this.sd[j]));
      z += v * this.beta[j];
    }
    // Level veto: no labelled gulp is louder than 40 dB above the floor (212 gulps), while loud
    // speech onsets reach 50+ dB. The linear score keeps rising with level, so it is capped here.
    let pk = -Infinity;
    for (let j = 0; j < this.vetoFrames; j++) pk = Math.max(pk, this.total[Math.min(ka + j, this.k) % R]);
    let ds = false;
    for (let j = 0; j < this.dsAhead && ka + j <= this.k; j++) if (this.dig[(ka + j) % R]) { ds = true; break; }
    const veto = pk - fl > this.vetoDb || ds ? 1 : this.breathRule && this._breath(a.k0) ? 2 : 0;
    const fired = z > this.eta && !veto;
    const tFire = this.tstart[ka % R];
    this.events.push([tFire, a.kind, z, fired, veto]);
    const t0 = Math.max(0, (this.fromOnset && a.kind === 1 ? this.tstart[a.k0 % R] : tFire) - this.preRoll);
    if (this.events.length > 5000) this.events.splice(0, 1000);
    if (!fired) return;
    this.nMutes += 1;
    if (tFire < (this.sInBlock - this.delay) / this.d.FS_IN) this.late += 1;
    // A firing anchor inside (or just after) an open mute extends it; whether it does is decided
    // at the anchor's own time. In read-ahead mode it may also move the mute's start back.
    const open = this.mutes.find(m => m.end === null || m.end > tFire);
    if (open && open.start <= tFire) {
      open.quiet = 0;
      if (open.end !== null && open.end < tFire + 0.025) { open.end = null; open.seen = this.k; }
      open.start = Math.min(open.start, t0);
      return;
    }
    const m = { start: t0, end: null, floor: fl, quiet: 0, speech: 0, k: ka, seen: this.k };
    for (let j = ka; j <= this.k; j++) {
      this._muteStep(m, j);
      if (m.end !== null) break;
    }
    this.mutes.push(m);
    this.muteLog.push(m);
    if (this.muteLog.length > 5000) this.muteLog.splice(0, 1000);
  }

  // inputs/outputs: arrays of Float32Array, one per channel, same length (128 in a worklet).
  process(inputs, outputs) {
    const n = inputs[0].length, ch = this.channels, d = this.d;
    this.sInBlock = this.sIn;
    const mono = new Float64Array(n);
    for (let c = 0; c < ch; c++) {
      const src = inputs[Math.min(c, inputs.length - 1)];
      for (let j = 0; j < n; j++) mono[j] += src[j] / ch;
    }
    const y = this.rs.push(mono);
    for (const v of y) this.y16.push(v);
    const W = d.WIN, Hp = d.HOP, FS16 = d.FS_IN / d.DOWN;
    while (Math.round(this.pos) + W - this.yOff <= this.y16.length) {
      const s0 = Math.round(this.pos), a = s0 - this.yOff;
      this.tstart[this.k % this.R] = s0 / FS16;       // y[n] lines up with input time n / 16 kHz
      this._frame(Float32Array.from(this.y16.slice(a, a + W)));
      this.pos += Hp / this.rate;
    }
    const drop = Math.round(this.pos) - this.yOff;
    if (drop > 0) { this.y16.splice(0, drop); this.yOff += drop; }
    this.sIn += n;
    if (!outputs) {
      // Analysis only (the read-ahead path in ahead.js): decisions and mute intervals, no delay
      // line and no gain. Old mutes are pruned exactly as below.
      const tEnd = (this.sIn - 1) / d.FS_IN;
      this.mutes = this.mutes.filter(m => m.end === null || m.end > tEnd - 1.0);
      return null;
    }

    for (let c = 0; c < ch; c++) {
      const src = inputs[Math.min(c, inputs.length - 1)], f = this.fifo[c];
      for (let j = 0; j < n; j++) f[(this.fifoW + j) % this.fifoLen] = src[j];
    }
    this.fifoW = (this.fifoW + n) % this.fifoLen;
    const tau0 = (this.sInBlock - this.delay) / d.FS_IN;
    let cur = this.gain;
    const g = new Float32Array(n);
    for (let j = 0; j < n; j++) {
      const tau = tau0 + j / d.FS_IN;
      let target = 1;
      if (!this.bypass) {
        for (const m of this.mutes) {
          if (tau >= m.start && (m.end === null || tau < m.end)) { target = this.gMute; break; }
        }
      }
      cur += Math.max(-this.step, Math.min(this.step, target - cur));
      g[j] = cur;
    }
    this.gain = cur;
    for (let c = 0; c < ch; c++) {
      const f = this.fifo[c], out = outputs[c];
      for (let j = 0; j < n; j++) out[j] = f[(this.fifoR + j) % this.fifoLen] * g[j];
    }
    this.fifoR = (this.fifoR + n) % this.fifoLen;
    const tLast = tau0 + (n - 1) / d.FS_IN;
    this.mutes = this.mutes.filter(m => m.end === null || m.end > tLast - 1.0);
    return g;
  }
}

const BREATH_FRAMES = 40, BREATH_LF_BANDS = 19;   // 400 ms; bands 0..18 end at 3203 Hz

function percentile(a, q) {
  // numpy's default 'linear' percentile.
  const s = Float64Array.from(a).sort();
  const pos = (s.length - 1) * q / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
