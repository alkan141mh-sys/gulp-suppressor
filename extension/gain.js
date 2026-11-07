// Read-ahead playback gain, used by worklet.js. Pure JavaScript (tested under Node).
//
// In read-ahead mode the audio is NOT delayed: the mute intervals are already known, in media
// time, before the audio arrives. For each sample the worklet needs that sample's media time:
//   m(T) = anchor.media + (T - anchor.ctx) * anchor.rate + offset
// where T is the AudioContext time, the anchor is a simultaneous reading of ctx.currentTime and
// video.currentTime, and offset is measured by the Calibrator (ahead_core.js).
// The gain ramps (RAMP_MS) so that it reaches the mute level exactly at the mute's start.
export class AheadGain {
  constructor(sr, rampMs, attenDb) {
    this.sr = sr;
    this.rampS = rampMs / 1000;
    this.step = 1 / (this.rampS * sr);
    this.setAtten(attenDb);
    this.mutes = [];         // sorted, merged [start, end] in media time
    this.anchor = null;      // { ctx, media, rate, playing }
    this.offset = 0;
    this.gain = 1;
    this.bypass = false;
    this.applied = 0;        // mutes that actually lowered the gain
    this.lastMute = null;    // start of the last mute counted
  }

  setAtten(db) { this.gMute = Math.pow(10, -db / 20); }
  setMutes(list) { this.mutes = list; }

  mediaAt(T) {
    const a = this.anchor;
    return a.media + (T - a.ctx) * a.rate + this.offset;
  }

  // ins/outs: arrays of Float32Array; T: context time of the block's first sample.
  // Returns the block's mean input power (for the Calibrator).
  process(ins, outs, T) {
    const n = ins[0].length, ch = outs.length;
    let p = 0;
    for (let c = 0; c < ins.length; c++) for (let j = 0; j < n; j++) p += ins[c][j] * ins[c][j];
    p /= n * ins.length;
    const a = this.anchor, active = a && a.playing && !this.bypass;
    let i = 0;
    let m0 = 0, dm = 0, lead = 0;
    if (active) {
      m0 = this.mediaAt(T); dm = a.rate / this.sr; lead = this.rampS * a.rate;
      // first interval that can matter for this block
      const ms = this.mutes;
      let lo = 0, hi = ms.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (ms[mid][1] <= m0) lo = mid + 1; else hi = mid; }
      i = lo;
    }
    let g = this.gain;
    for (let j = 0; j < n; j++) {
      let target = 1;
      if (active) {
        const m = m0 + j * dm;
        while (i < this.mutes.length && this.mutes[i][1] <= m) i++;
        if (i < this.mutes.length && m + lead >= this.mutes[i][0]) {
          target = this.gMute;
          if (this.mutes[i][0] !== this.lastMute && m >= this.mutes[i][0]) { this.lastMute = this.mutes[i][0]; this.applied++; }
        }
      }
      g += Math.max(-this.step, Math.min(this.step, target - g));
      for (let c = 0; c < ch; c++) outs[c][j] = ins[Math.min(c, ins.length - 1)][j] * g;
    }
    this.gain = g;
    return p;
  }
}
