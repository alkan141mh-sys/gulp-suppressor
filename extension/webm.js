// Streaming WebM (Matroska) demuxer for the audio YouTube appends to its SourceBuffer.
//
// YouTube hands MSE an init segment (EBML header, Segment > Info, Tracks) and then media
// segments (Clusters of SimpleBlocks), in appendBuffer() calls whose boundaries need not match
// element boundaries. This parser keeps a byte queue and walks the elements as a flat stream:
// the masters it needs (Segment, Info, Tracks, TrackEntry, Audio, Cluster, BlockGroup) are
// entered by consuming only their header; every other element is skipped by its size; the few
// leaves it needs are read once completely buffered. Element IDs are unique across levels, so
// no stack is needed. Pure JavaScript: also runs under Node (extension/test/webm_test.mjs).
//
// Output, per audio frame: { track, time (s, presentation time), data (Uint8Array) }.

const ID = {
  EBML: 0x1A45DFA3, Segment: 0x18538067, Info: 0x1549A966, TimecodeScale: 0x2AD7B1,
  Tracks: 0x1654AE6B, TrackEntry: 0xAE, TrackNumber: 0xD7, TrackType: 0x83, CodecID: 0x86,
  CodecPrivate: 0x63A2, CodecDelay: 0x56AA, Audio: 0xE1, SamplingFrequency: 0xB5, Channels: 0x9F,
  Cluster: 0x1F43B675, Timecode: 0xE7, SimpleBlock: 0xA3, BlockGroup: 0xA0, Block: 0xA1,
};
const ENTER = new Set([ID.Segment, ID.Info, ID.Tracks, ID.TrackEntry, ID.Audio, ID.Cluster, ID.BlockGroup]);
const READ = new Set([ID.TimecodeScale, ID.TrackNumber, ID.TrackType, ID.CodecID, ID.CodecPrivate,
                      ID.CodecDelay, ID.SamplingFrequency, ID.Channels, ID.Timecode, ID.SimpleBlock, ID.Block]);
const MAX_LEAF = 1 << 24;     // a leaf this large means we lost sync

export class WebmDemuxer {
  constructor() {
    this.buf = new Uint8Array(0);
    this.off = 0;
    this.skip = 0;
    this.scale = 1e6;          // TimecodeScale, ns per tick
    this.tracks = [];
    this.cluster = null;       // Timecode of the current cluster, in ticks
    this.errors = 0;
    this.lacedSkipped = 0;
    this.ids = new Map();
    this.framesAll = 0;
    this.noCluster = 0;
  }

  // Forget a partly parsed element (SourceBuffer.abort(): the next append starts afresh).
  // Track information is kept: YouTube does not resend the init segment after a seek.
  abort() { this.buf = new Uint8Array(0); this.off = 0; this.skip = 0; }

  audioTrack() { return this.tracks.find(t => t.type === 2) || null; }

  push(bytes) {
    // An append that starts with a Cluster or an EBML header starts at an element boundary
    // (after a seek the player appends from a new place without telling us). If the parser is
    // still inside an unfinished element from the old place, drop that element.
    const id4 = bytes.length >= 4 ? ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]) >>> 0 : 0;
    if ((id4 === ID.Cluster || id4 === ID.EBML) && (this.skip || this.off < this.buf.length)) {
      this.resyncs = (this.resyncs || 0) + 1;
      this.abort();
    }
    this.lastHead = Array.from(bytes.subarray(0, 4), b => b.toString(16).padStart(2, '0')).join('');
    const rest = this.buf.subarray(this.off);
    const b = new Uint8Array(rest.length + bytes.length);
    b.set(rest); b.set(bytes, rest.length);
    this.buf = b; this.off = 0;
    const frames = [];
    for (;;) {
      if (this.skip) {
        const k = Math.min(this.skip, this.buf.length - this.off);
        this.off += k; this.skip -= k;
        if (this.skip) break;
      }
      const h = this._header(this.off);
      if (!h) break;                                    // header not complete yet
      if (h.bad) { this._lost(); break; }
      const { id, size, len } = h;
      this.ids.set(id, (this.ids.get(id) || 0) + 1);       // diagnostics: elements seen
      if (ENTER.has(id)) {
        this.off += len;
        if (id === ID.TrackEntry) this.tracks.push({ number: 0, type: 0, codec: '', private: null, rate: 0, channels: 1, delay: 0 });
        if (id === ID.Cluster) this.cluster = null;
        continue;
      }
      if (size === null) { this._lost(); break; }       // only masters may have unknown size
      if (READ.has(id)) {
        if (size > MAX_LEAF) { this._lost(); break; }
        if (this.buf.length - this.off - len < size) break;   // wait for the rest
        const p = this.buf.subarray(this.off + len, this.off + len + size);
        this.off += len + size;
        this._leaf(id, p, frames);
        continue;
      }
      this.off += len;
      this.skip = size;
    }
    return frames;
  }

  _lost() {
    // Out of sync (should not happen with whole appends); drop what is queued.
    this.errors++;
    this.abort();
  }

  _vint(o, keepMarker) {
    // EBML variable-length integer at buf[o]: returns { value, len } or null if incomplete.
    const b = this.buf;
    if (o >= b.length) return null;
    const first = b[o];
    if (first === 0) return { bad: true };
    let len = 1;
    while (!(first & (0x80 >> (len - 1)))) len++;
    if (len > 8) return { bad: true };
    if (o + len > b.length) return null;
    let v = keepMarker ? first : first & (0xFF >> len);
    let allOnes = v === (0xFF >> len);
    for (let i = 1; i < len; i++) {
      v = v * 256 + b[o + i];
      if (b[o + i] !== 0xFF) allOnes = false;
    }
    return { value: v, len, unknown: !keepMarker && allOnes };
  }

  _header(o) {
    const id = this._vint(o, true);
    if (!id) return null;
    if (id.bad || id.len > 4) return { bad: true };
    const sz = this._vint(o + id.len, false);
    if (!sz) return null;
    if (sz.bad) return { bad: true };
    return { id: id.value, size: sz.unknown ? null : sz.value, len: id.len + sz.len };
  }

  _leaf(id, p, frames) {
    const uint = () => { let v = 0; for (const x of p) v = v * 256 + x; return v; };
    const t = this.tracks[this.tracks.length - 1];
    switch (id) {
      case ID.TimecodeScale: this.scale = uint(); break;
      case ID.TrackNumber: if (t) t.number = uint(); break;
      case ID.TrackType: if (t) t.type = uint(); break;
      case ID.CodecID: if (t) t.codec = String.fromCharCode(...p); break;
      case ID.CodecPrivate: if (t) t.private = p.slice(); break;
      case ID.CodecDelay: if (t) t.delay = uint() / 1e9; break;
      case ID.Channels: if (t) t.channels = uint(); break;
      case ID.SamplingFrequency: {
        const dv = new DataView(p.buffer, p.byteOffset, p.byteLength);
        if (t) t.rate = p.length === 4 ? dv.getFloat32(0) : dv.getFloat64(0);
        break;
      }
      case ID.Timecode: this.cluster = uint(); break;
      case ID.SimpleBlock: case ID.Block: this._block(p, frames); break;
    }
  }

  _block(p, frames) {
    if (this.cluster === null) { this.noCluster++; return; }   // block before its cluster's timecode
    let n = p[0], len = 1;
    while (!(n & (0x80 >> (len - 1)))) len++;
    let track = p[0] & (0xFF >> len);
    for (let i = 1; i < len; i++) track = track * 256 + p[i];
    const rel = (p[len] << 24 >> 16) | p[len + 1];      // signed 16-bit, big endian
    const flags = p[len + 2];
    const lacing = (flags >> 1) & 3;
    const time = (this.cluster + rel) * this.scale / 1e9;
    if (lacing) { this.lacedSkipped++; return; }        // YouTube's Opus is never laced
    this.framesAll++;
    frames.push({ track, time, data: p.slice(len + 3) });
  }
}
