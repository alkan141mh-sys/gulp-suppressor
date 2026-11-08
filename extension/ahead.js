// Read-ahead controller (browser side): turns the bytes page.js copies from YouTube's audio
// SourceBuffer into mute intervals in media time, and keeps the worklet's clock aligned.
//
//   page.js --postMessage--> demux (webm.js) --> AudioDecoder (WebCodecs, Opus)
//     --> Analyzer (ahead_core.js: engine.js per contiguous stretch) --> mutes in media time
//     --> worklet (gain.js applies them with no delay)
//   worklet input levels --> Calibrator --> clock offset --> worklet
import { WebmDemuxer } from './webm.js';
import { Analyzer, Calibrator, FS } from './ahead_core.js';

const SEND_AHEAD_S = 120;        // mutes sent to the worklet: from 2 s behind to this far ahead

export class AheadController {
  constructor({ video, ctx, node, model, settings, log }) {
    Object.assign(this, { video, ctx, node, model, log });
    this.settings = { ...settings };
    this.analyzer = new Analyzer(model, settings.fa);
    this.cal = new Calibrator();
    this.sb = null;              // current audio SourceBuffer: { id, type, demux, decoder, ok }
    this.on = false;             // worklet in read-ahead mode
    this.anchor = null;
    this.sentKey = '';
    this.working = false;
    this.stats = { appends: 0, frames: 0, decoded: 0, decodeErrors: 0, anchors: 0, calibrations: 0, gaps: 0 };
    this.chan = new MessageChannel();               // yields to the page between analysis slices
    this.chan.port1.onmessage = () => this._work();
    for (const ev of ['playing', 'pause', 'seeked', 'ratechange', 'waiting'])
      video.addEventListener(ev, () => this._anchor(ev));
    video.addEventListener('loadstart', () => this._newVideo());
    this.timer = setInterval(() => this._tick(), 250);
  }

  // ---- input from page.js ----
  onPageMessage(d) {
    if (d.kind === 'sb') {
      // Another element's stream (a hover preview): not ours.
      if (d.url && this.video.src && this.video.src !== d.url) return;
      this._newVideo();
      const webm = /^audio\/webm/i.test(d.type) && /opus/i.test(d.type);
      this.sb = { id: d.id, type: d.type, demux: webm ? new WebmDemuxer() : null, decoder: null, track: null };
      if (!webm) this.log(`read-ahead unavailable for ${d.type}; live mode`);
      return;
    }
    const sb = this.sb;
    if (!sb || d.id !== sb.id || !sb.demux) return;
    if (d.kind === 'abort') { this.stats.aborts = (this.stats.aborts || 0) + 1; sb.demux.abort(); return; }
    if (d.kind !== 'append' || !(d.bytes instanceof ArrayBuffer)) return;
    this.stats.appends++;
    const u8 = new Uint8Array(d.bytes);
    (this.appendLog ||= []).push(`${u8.length}B ${Array.from(u8.subarray(0, 6), b => b.toString(16).padStart(2, '0')).join('')} tso${d.tso}`);
    if (this.appendLog.length > 4) this.appendLog.shift();
    const frames = sb.demux.push(u8);
    if (frames.length) {
      const f0 = frames[0], f1 = frames[frames.length - 1];
      this.lastFrames = `${frames.length} kare iz ${f0.track} ${f0.time.toFixed(2)}–${f1.time.toFixed(2)} s`;
    }
    const tr = sb.demux.audioTrack();
    if (!tr || !frames.length) return;
    if (!sb.decoder) this._decoder(sb, tr);
    if (!sb.decoder) return;
    const tso = Number.isFinite(d.tso) ? d.tso : 0;
    for (const f of frames) {
      if (f.track !== tr.number) continue;
      this.stats.frames++;
      // Chrome's AudioDecoder stamps its output from the FIRST input timestamp plus the samples
      // decoded so far; later input timestamps are ignored. So after a jump (the player appending
      // from a new place after a seek) a fresh decoder is needed, or the audio lands at the wrong
      // time. Opus frames here are 20 ms apart; a larger gap or a step back is a jump.
      const t = f.time + tso;
      if (sb.lastT !== undefined && (t - sb.lastT > 0.1 || t < sb.lastT)) {
        this.stats.restarts = (this.stats.restarts || 0) + 1;
        const old = sb.decoder;
        sb.decoder = null;
        if (old) old.flush().then(() => old.close(), () => {});
        this._decoder(sb, tr);
        if (!sb.decoder) return;
      }
      sb.lastT = t;
      try {
        sb.decoder.decode(new EncodedAudioChunk({ type: 'key', timestamp: Math.round((f.time + tso) * 1e6), data: f.data }));
      } catch (e) { this.stats.decodeErrors++; }
    }
  }

  _decoder(sb, tr) {
    if (tr.codec !== 'A_OPUS' || typeof AudioDecoder === 'undefined') {
      this.log(`read-ahead unavailable (codec ${tr.codec}); live mode`);
      sb.demux = null;
      return;
    }
    const config = { codec: 'opus', sampleRate: tr.rate || 48000, numberOfChannels: tr.channels || 2 };
    if (tr.private) config.description = tr.private;
    const dec = new AudioDecoder({
      output: (ad) => this._decoded(ad),
      error: (e) => { this.stats.decodeErrors++; this.log('decoder error', e); if (sb.decoder === dec) sb.decoder = null; },
    });
    dec.configure(config);
    sb.decoder = dec;
    this.log(`read-ahead: Opus ${config.numberOfChannels} ch @ ${config.sampleRate} Hz`);
  }

  _decoded(ad) {
    try {
      if (ad.sampleRate !== FS) { this.stats.decodeErrors++; return; }
      const n = ad.numberOfFrames, ch = ad.numberOfChannels;
      const mono = new Float32Array(n), tmp = new Float32Array(n);
      for (let c = 0; c < ch; c++) {
        ad.copyTo(tmp, { planeIndex: c, format: 'f32-planar' });
        for (let i = 0; i < n; i++) mono[i] += tmp[i] / ch;
      }
      this.analyzer.add(ad.timestamp / 1e6, mono);
      this.stats.decoded += n;
      if (!this.on) this._setOn(true);
      this._kick();
    } finally {
      ad.close();
    }
  }

  // ---- analysis, in slices so the page stays responsive ----
  _kick() { if (!this.working) { this.working = true; this.chan.port2.postMessage(0); } }
  _work() {
    const more = this.analyzer.step(6);
    this._send();
    if (more) this.chan.port2.postMessage(0); else this.working = false;
  }

  _send(force = false) {
    if (!this.on) return;
    const t = this.video.currentTime;
    const mutes = this.analyzer.mutes(t - 2, t + SEND_AHEAD_S);
    const key = JSON.stringify(mutes);
    if (!force && key === this.sentKey) return;
    this.sentKey = key;
    this.node.port.postMessage({ type: 'mutes', mutes });
  }

  // ---- clocks ----
  _anchor(why) {
    const v = this.video;
    this.anchor = { ctx: this.ctx.currentTime, media: v.currentTime, rate: v.playbackRate || 1,
                    playing: !v.paused && !v.seeking && v.readyState >= 3 && why !== 'waiting' && why !== 'pause' };
    this.stats.anchors++;
    this.cal.newAnchor();
    this.node.port.postMessage({ type: 'anchor', anchor: this.anchor });
    if (why === 'seeked') this._send(true);
  }

  onLevels(pairs) {
    this.cal.add(pairs);
    const now = this.ctx.currentTime;
    // Every 0.3 s while the anchor is young (a seek, rate change or resume), then every second.
    const every = now - this.anchor?.ctx < 3 ? 0.3 : 1;
    if (this.on && this.anchor && this.anchor.playing && (!this.lastCal || now - this.lastCal > every)) {
      this.lastCal = now;
      const est = this.cal.estimate(this.anchor, this.analyzer);
      if (this.cal.update(est)) {
        this.stats.calibrations++;
        this.node.port.postMessage({ type: 'offset', offset: this.cal.offset });
      }
    }
  }

  _tick() {
    const v = this.video;
    if (!this.on) return;
    // Re-anchor when the element's clock and the prediction disagree (stall, a seek without
    // events, playing after buffering) or when the playing state changed.
    const playing = !v.paused && !v.seeking && v.readyState >= 3;
    const a = this.anchor;
    if (!a || a.playing !== playing || Math.abs(a.media + (this.ctx.currentTime - a.ctx) * a.rate - v.currentTime) > 0.06 * a.rate + 0.03)
      this._anchor('tick');
    this.analyzer.evict(v.currentTime);
    const cov = this.analyzer.coveredUntil(v.currentTime);
    if (playing && (cov === null || cov < v.currentTime)) this.stats.gaps++;
    this._send();
  }

  _setOn(on) {
    this.on = on;
    this.node.port.postMessage({ type: 'ahead', on });
    if (on) { this._anchor('on'); this._send(true); this.node.port.postMessage({ type: 'offset', offset: this.cal.offset }); }
    this.log(on ? 'read-ahead mode on' : 'read-ahead mode off');
  }

  _newVideo() {
    if (this.sb && this.sb.decoder) { try { this.sb.decoder.close(); } catch (e) { /* closed */ } }
    this.sb = null;
    this.analyzer = new Analyzer(this.model, this.settings.fa);
    this.cal = new Calibrator();
    this.sentKey = '';
    if (this.on) this._setOn(false);
  }

  // Not a Buders video: drop the stream and the analysis; page.js sends nothing until a Buders
  // video plays again (its 'sb' message starts a new stream).
  setActive(on) {
    if (!on) this._newVideo();
  }

  setSettings(s) {
    this.settings = { ...s };
    this.analyzer.setFa(s.fa);
    this._kick();
  }

  status() {
    const t = this.video.currentTime, cov = this.analyzer.coveredUntil(t);
    const near = [...this.analyzer.segs].sort((a, b) => Math.abs((a.t0 + a.end()) / 2 - t) - Math.abs((b.t0 + b.end()) / 2 - t))
      .slice(0, 3).map(s => `${s.t0.toFixed(1)}–${s.end().toFixed(1)}`);
    return { ahead: this.on, aheadS: cov === null ? null : +(cov - t).toFixed(1),
             t: +t.toFixed(1), near, sbType: this.sb ? this.sb.type : null, hasDecoder: !!(this.sb && this.sb.decoder),
             duplicates: this.analyzer.duplicates, demuxErrors: this.sb && this.sb.demux ? this.sb.demux.errors : null,
             demuxState: this.sb && this.sb.demux ? `skip ${this.sb.demux.skip}, bekleyen ${this.sb.demux.buf.length - this.sb.demux.off} B, ` +
               `resync ${this.sb.demux.resyncs || 0}, abort ${this.stats.aborts || 0}, tüm kare ${this.sb.demux.framesAll}, ` +
               `kümesiz blok ${this.sb.demux.noCluster}, izler ${this.sb.demux.tracks.map(t => `${t.number}/${t.type}/${t.codec}`).join(' ')}, ` +
               `öğeler ${[...this.sb.demux.ids].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k, v]) => `${k.toString(16)}×${v}`).join(' ')}, ` +
               `son kareler ${this.lastFrames}, son eklemeler ${(this.appendLog || []).join(' | ')}` : null,
             offsetMs: +(this.cal.offset * 1000).toFixed(1), calR: this.cal.last ? +this.cal.last.r.toFixed(3) : null,
             segments: this.analyzer.segs.length, ...this.stats };
  }
}
