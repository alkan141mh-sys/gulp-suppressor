// AudioWorklet wrapper. Runs on the browser's audio thread, 128 samples per call at 48 kHz
// (the content script creates the AudioContext at 48 kHz, so the model's sample rate always
// holds; Chrome resamples to the output device).
//
// Two modes:
//   live   engine.js detects in the stream itself; the audio is delayed by DELAY_MS so a
//          decision can still mute the gulp it was made on.
//   ahead  the content script has already found the gulps in the audio YouTube buffered
//          (ahead.js); this node only applies those mutes (gain.js), with no delay. It also
//          reports its input level per block so the content script can align the clocks.
import { Engine } from './engine.js';
import { AheadGain } from './gain.js';

class GulpProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.model = o.model;
    this.engine = null;
    this.make(o);
    if (o.rate) this.engine.setRate(o.rate);
    this.ahead = false;
    this.ag = new AheadGain(sampleRate, o.model.dsp.RAMP_MS, o.atten);
    this.ag.bypass = !o.enabled;
    this.levels = new Float64Array(64); this.nl = 0;   // [T, power] pairs for the Calibrator
    this.lastReport = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'settings') {
        this.engine.bypass = !m.enabled;
        this.engine.setOperatingPoint(this.model, m.fa);
        this.engine.setAttenuation(m.atten);
        this.ag.bypass = !m.enabled;
        this.ag.setAtten(m.atten);
      } else if (m.type === 'rate') {
        this.engine.setRate(m.rate);
      } else if (m.type === 'reset') {
        // Seek, play after pause, or a new video: the stream restarts. Within a video the room
        // floor and speech level carry over; a new video starts them from scratch.
        this.make({ fa: m.fa, atten: m.atten, enabled: m.enabled,
                    carry: m.newVideo ? null : this.engine.carryState() });
        if (m.rate) this.engine.setRate(m.rate);
      } else if (m.type === 'ahead') {
        this.ahead = !!m.on;
        this.ag.gain = 1;
      } else if (m.type === 'anchor') {
        this.ag.anchor = m.anchor;
      } else if (m.type === 'offset') {
        this.ag.offset = m.offset;
      } else if (m.type === 'mutes') {
        this.ag.setMutes(m.mutes);
      } else if (m.type === 'dump') {
        this.port.postMessage({ type: 'events', events: this.engine.events, late: this.engine.late,
                                nMutes: this.engine.nMutes });
      }
    };
  }

  make(o) {
    const n = this.engine ? this.engine.nMutes : 0;
    this.engine = new Engine(this.model, o.fa, o.atten, 2, o.carry || null);
    this.engine.nMutes = n;                  // the popup counts mutes per page, across resets
    this.engine.bypass = !o.enabled;
    this.zero = new Float32Array(128);
  }

  process(inputs, outputs) {
    const inp = inputs[0], out = outputs[0];
    if (!out || out.length === 0) return true;
    // Paused or no input: feed silence so time, the delay line and the trackers keep running.
    const n = out[0].length;
    if (this.zero.length !== n) this.zero = new Float32Array(n);
    const ins = inp && inp.length ? (inp.length === 1 ? [inp[0], inp[0]] : inp) : [this.zero, this.zero];
    const outs = out.length === 1 ? [out[0], new Float32Array(n)] : out;
    if (this.ahead) {
      const p = this.ag.process(ins, outs, currentTime);
      this.levels[this.nl++] = currentTime; this.levels[this.nl++] = p;
      if (this.nl === this.levels.length) {
        this.port.postMessage({ type: 'levels', pairs: this.levels }, [this.levels.buffer]);
        this.levels = new Float64Array(64); this.nl = 0;
      }
    } else {
      this.engine.process(ins, outs);
    }
    if (currentTime - this.lastReport > 0.5) {
      this.lastReport = currentTime;
      this.port.postMessage({ type: 'stats', nMutes: this.engine.nMutes + this.ag.applied, late: this.engine.late });
    }
    return true;
  }
}

registerProcessor('gulp-suppressor', GulpProcessor);
