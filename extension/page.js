// Runs in YouTube's own JavaScript world (manifest: "world": "MAIN", document_start), before
// the player's scripts. It copies every byte the player appends to an AUDIO SourceBuffer and
// posts the copy to the extension's content script (content.js / ahead.js), which decodes it
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

  const ids = new WeakMap();
  let next = 1;
  const addSourceBuffer = MS.prototype.addSourceBuffer;
  MS.prototype.addSourceBuffer = function (type) {
    const sb = addSourceBuffer.call(this, type);
    if (typeof type === 'string' && /^audio\//i.test(type)) {
      const id = next++;
      ids.set(sb, id);
      post({ kind: 'sb', id, type, url: urls.get(this) || null });
    }
    return sb;
  };

  const proto = window.SourceBuffer.prototype;
  const appendBuffer = proto.appendBuffer;
  proto.appendBuffer = function (data) {
    const id = ids.get(this);
    if (id) {
      try {
        const u = data instanceof ArrayBuffer ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const copy = u.slice();
        post({ kind: 'append', id, tso: this.timestampOffset, bytes: copy.buffer }, [copy.buffer]);
      } catch (e) { /* ignore */ }
    }
    return appendBuffer.call(this, data);
  };
  // Which channel the watch page's video belongs to (content.js runs only on Buders videos).
  // The player's own response is read once it describes the video in the address bar; on the
  // way to a new page the channel is cleared at once, so nothing is muted before it is known.
  let tries = 0, timer = 0;
  const channel = () => {
    clearTimeout(timer);
    try {
      const v = location.pathname === '/watch' ? new URLSearchParams(location.search).get('v') : null;
      if (!v) { post({ kind: 'channel', id: null, videoId: null }); return; }
      const p = document.getElementById('movie_player');
      const vd = [p && p.getPlayerResponse && p.getPlayerResponse(), window.ytInitialPlayerResponse]
        .map(r => r && r.videoDetails).find(d => d && d.videoId === v);
      if (vd) { post({ kind: 'channel', id: vd.channelId || null, videoId: v }); return; }
      if (tries++ < 60) timer = setTimeout(channel, 250);       // the player has not caught up yet
    } catch (e) { /* ignore */ }
  };
  const recheck = () => { tries = 0; channel(); };
  document.addEventListener('yt-navigate-start', () => post({ kind: 'channel', id: null, videoId: null }));
  document.addEventListener('yt-navigate-finish', recheck);
  document.addEventListener('yt-player-updated', recheck);
  document.addEventListener('DOMContentLoaded', recheck);
  window.addEventListener('popstate', recheck);

  const abort = proto.abort;
  proto.abort = function () {
    const id = ids.get(this);
    if (id) post({ kind: 'abort', id });
    return abort.call(this);
  };
})();
