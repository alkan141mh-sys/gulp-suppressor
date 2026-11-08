// Runs in YouTube's own JavaScript world (manifest: "world": "MAIN", document_start), before
// the player's scripts. On Buders videos it copies every byte the player appends to an AUDIO
// SourceBuffer and posts the copy to the extension's content script (content.js / ahead.js), which decodes it
// and finds gulps seconds before they play. The player's own calls are passed through
// unchanged; if anything here fails, the player works exactly as without the extension.
(() => {
  if (window.__gulpHook) return;
  window.__gulpHook = true;
  const MS = window.MediaSource;
  if (!MS || !window.SourceBuffer) return;

  const post = (msg, transfer) => {
    try { window.postMessage({ __gulp: 1, ...msg }, location.origin, transfer || []); } catch (e) { /* ignore */ }
  };

  // Ask for Opus. The player picks its audio format with isTypeSupported(); answering no for
  // AAC (audio/mp4) makes it choose WebM/Opus, which the content script can demux and decode.
  const isSupported = MS.isTypeSupported.bind(MS);
  MS.isTypeSupported = function (type) {
    if (typeof type === 'string' && /^audio\/mp4/i.test(type) && isSupported('audio/webm; codecs="opus"')) return false;
    return isSupported(type);
  };

  // Which element plays which MediaSource: the player attaches it as video.src = blob URL.
  // The content script follows only the main player, not the muted hover previews.
  const urls = new WeakMap();
  const createObjectURL = URL.createObjectURL;
  URL.createObjectURL = function (obj) {
    const url = createObjectURL.call(URL, obj);
    try { if (obj instanceof MS) urls.set(obj, url); } catch (e) { /* ignore */ }
    return url;
  };

  // Copies are made only for Buders videos. Each audio SourceBuffer is a stream in one of three
  // states:
  //   wait  the video's channel is not known yet: copies are held here (bounded), not posted
  //   on    a Buders video: every append is copied and posted
  //   off   another channel: nothing is copied (apart from the stream's first append, its WebM
  //         header, in case the same SourceBuffer later plays a Buders video)
  // Keep CHANNELS in step with content.js.
  const CHANNELS = ['UClKBd-4xFVodFmK3bZfJAAQ'];
  const HOLD_BYTES = 16 << 20;           // at most ~10 minutes of Opus held while waiting
  let chan = null;                       // null: unknown, true: Buders, false: another channel
  const ids = new WeakMap();
  const streams = new Map();             // id -> { state, held: [[msg, transfer]], size, sb, init }
  let next = 1;

  const bytesOf = (data) => data instanceof ArrayBuffer ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const emit = (s, msg, tr) => {
    if (s.state === 'on') post(msg, tr);
    else if (s.state === 'wait') {
      s.held.push([msg, tr]);
      s.size += msg.bytes ? msg.bytes.byteLength : 0;
      while (s.size > HOLD_BYTES && s.held.length > 3) {          // keep 'sb' and the header, drop the oldest audio
        const [[m]] = s.held.splice(2, 1);
        s.size -= m.bytes ? m.bytes.byteLength : 0;
      }
    }
  };
  const setState = (s, state) => {
    if (s.state === state) return;
    if (state === 'on') {
      if (s.state === 'wait') for (const [m, tr] of s.held) post(m, tr);
      else {                                                      // off -> on: header, then live appends
        post(s.sb);
        if (s.init) { const c = s.init.slice(0); post({ ...s.initMsg, bytes: c }, [c]); }
      }
    }
    s.held = []; s.size = 0;
    s.state = state;
  };
  const setChannel = (id) => {           // id: a channel ID, null = not known yet, false = no video page
    chan = id === null ? null : id !== false && CHANNELS.includes(id);
    const all = [...streams.values()];
    for (const s of all) {
      // Buders: streams that waited, and the newest one even if it was off (a SourceBuffer the
      // player reuses for the new video); older ones belong to players and previews long gone.
      if (chan === true && (s.state === 'wait' || s === all[all.length - 1])) setState(s, 'on');
      else if (chan === false) setState(s, 'off');
    }
  };

  const addSourceBuffer = MS.prototype.addSourceBuffer;
  MS.prototype.addSourceBuffer = function (type) {
    const sb = addSourceBuffer.call(this, type);
    if (typeof type === 'string' && /^audio\//i.test(type)) {
      const id = next++;
      ids.set(sb, id);
      const s = { state: 'wait', held: [], size: 0, sb: { kind: 'sb', id, type, url: urls.get(this) || null }, init: null };
      streams.set(id, s);
      if (streams.size > 16) streams.delete(streams.keys().next().value);   // old players and previews
      if (chan === true) s.state = 'on';
      emit(s, s.sb);
      if (chan === false) setState(s, 'off');
    }
    return sb;
  };

  const proto = window.SourceBuffer.prototype;
  const appendBuffer = proto.appendBuffer;
  proto.appendBuffer = function (data) {
    const s = streams.get(ids.get(this));
    if (s) {
      try {
        const msg = { kind: 'append', id: s.sb.id, tso: this.timestampOffset };
        if (!s.init) { s.init = bytesOf(data).slice().buffer; s.initMsg = msg; }
        if (s.state !== 'off') {
          const copy = bytesOf(data).slice();
          emit(s, { ...msg, bytes: copy.buffer }, [copy.buffer]);
        }
      } catch (e) { /* ignore */ }
    }
    return appendBuffer.call(this, data);
  };
  const abort = proto.abort;
  proto.abort = function () {
    const s = streams.get(ids.get(this));
    if (s) emit(s, { kind: 'abort', id: s.sb.id });
    return abort.call(this);
  };

  // Which channel the watch page's video belongs to. The player's own response is read once it
  // describes the video in the address bar; on the way to a new page the channel is cleared at
  // once, so nothing is copied or muted before it is known.
  const announce = (id, videoId) => { post({ kind: 'channel', id: id || null, videoId }); setChannel(id); };
  let tries = 0, timer = 0;
  const channel = () => {
    clearTimeout(timer);
    try {
      const v = location.pathname === '/watch' ? new URLSearchParams(location.search).get('v') : null;
      if (!v) { announce(false, null); return; }          // home, search, ...: hover previews are not ours
      const p = document.getElementById('movie_player');
      const vd = [p && p.getPlayerResponse && p.getPlayerResponse(), window.ytInitialPlayerResponse]
        .map(r => r && r.videoDetails).find(d => d && d.videoId === v);
      if (vd) { announce(vd.channelId || false, v); return; }
      if (tries++ < 60) timer = setTimeout(channel, 250);       // the player has not caught up yet
    } catch (e) { /* ignore */ }
  };
  const recheck = () => { tries = 0; channel(); };
  document.addEventListener('yt-navigate-start', () => announce(null, null));
  document.addEventListener('yt-navigate-finish', recheck);
  document.addEventListener('yt-player-updated', recheck);
  document.addEventListener('DOMContentLoaded', recheck);
  window.addEventListener('popstate', recheck);
})();
