// Content script for youtube.com: routes the <video> element's sound through the suppressor.
//
//   <video> -> MediaElementSource -> gulp-suppressor (AudioWorklet) -> speakers
//
// Read-ahead mode (normal case): page.js copies the audio YouTube buffers; ahead.js decodes it
// and finds the gulps seconds before they play, and the worklet mutes them with no delay.
// Live mode (fallback, e.g. no Opus audio): the worklet detects in the stream itself, with the
// audio delayed 115 ms.
// YouTube plays media from same-origin blob: URLs, so a MediaElementSource receives the audio
// (cross-origin media without CORS would come out silent). If the page blocks loading the
// worklet module, the same engine runs in a ScriptProcessorNode instead (older API, on the main
// thread, ~10 ms more latency, live mode only). If both fail, nothing is captured and the video
// plays normally.
//
// Only Buders videos: the detector is trained on one lecturer, so on any other channel the
// extension takes no part. page.js reports the channel of the video on the watch page; the
// sound is taken over only once a Buders video plays, and if the same player then moves on to
// another channel it passes the sound through untouched (bypass) until a Buders video is back.
(() => {
  const CHANNELS = ['UClKBd-4xFVodFmK3bZfJAAQ'];   // BUders Boğaziçiliden Özel Ders (@Buders)
  const DEFAULTS = { enabled: true, fa: '2', atten: 30 };
  let settings = { ...DEFAULTS };
  let channel = { id: null, videoId: null };
  const allowed = () => CHANNELS.includes(channel.id);
  const eff = () => ({ ...settings, enabled: settings.enabled && allowed() });   // what the audio path uses
  let model = null;
  let current = null;            // { ctx, node, mode, stats, video, engine?, ahead? }
  const seen = new WeakSet();

  const log = (...a) => console.log('[gulp]', ...a);

  // Messages from page.js start before the player loads anything (document_start), so they are
  // collected from the first moment and handed over once the controller exists.
  let pageQueue = [];
  window.addEventListener('message', (e) => {
    const d = e.data;
    if (e.source !== window || !d || d.__gulp !== 1 || typeof d.kind !== 'string') return;
    if (d.kind === 'channel') {
      const was = allowed();
      channel = { id: typeof d.id === 'string' ? d.id : null, videoId: d.videoId || null };
      if (allowed() !== was) log(allowed() ? 'Buders video: active' : 'not a Buders video: sound passes through untouched');
      if (allowed() && !current) scan();
      apply();
      return;
    }
    if (current && current.ahead) current.ahead.onPageMessage(d);
    else if (pageQueue) {
      if (d.kind === 'sb') pageQueue = [];        // only the newest stream matters
      pageQueue.push(d);
      if (pageQueue.length > 2000) pageQueue.splice(1, 1000);
    }
  });

  async function init() {
    model = await (await fetch(chrome.runtime.getURL('model.json'))).json();
    settings = { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))) };
    const start = () => {
      scan();
      new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
    else start();
  }

  function scan() {
    // After the extension is reloaded, this copy of the script is orphaned (chrome.runtime.id is
    // gone) and can no longer load extension files: stay quiet until the page is refreshed.
    if (!chrome.runtime?.id) return;
    if (current || !model || !allowed()) return;          // one player per page; nothing on other channels
    const v = document.querySelector('video');
    if (v && !seen.has(v)) {
      seen.add(v);
      attach(v).catch((e) => log('could not attach, video plays unprocessed:', e));
    }
  }

  function newEngine(Engine, carry = null) {
    const e = new Engine(model, settings.fa, settings.atten, 2, carry);
    e.bypass = !eff().enabled;
    return e;
  }

  function scriptProcessor(ctx, holder) {
    // holder.engine can be swapped (reset on a new video); the callback always uses the current one.
    const node = ctx.createScriptProcessor(512, 2, 2);
    node.onaudioprocess = (e) => {
      const ib = e.inputBuffer, ob = e.outputBuffer;
      holder.engine.process([ib.getChannelData(0), ib.getChannelData(1)], [ob.getChannelData(0), ob.getChannelData(1)]);
    };
    return node;
  }

  async function attach(video) {
    const ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
    let node, mode, Engine = null, ahead = null;
    const holder = { engine: null };
    try {
      await ctx.audioWorklet.addModule(chrome.runtime.getURL('worklet.js'));
      node = new AudioWorkletNode(ctx, 'gulp-suppressor', {
        numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2],
        channelCount: 2, channelCountMode: 'explicit',
        processorOptions: { model, fa: settings.fa, atten: settings.atten, enabled: eff().enabled,
                            rate: video.playbackRate },
      });
      mode = 'AudioWorklet';
      try {
        const { AheadController } = await import(chrome.runtime.getURL('ahead.js'));
        ahead = new AheadController({ video, ctx, node, model, settings, log });
      } catch (e) {
        log('read-ahead unavailable, live mode only:', e);
      }
    } catch (e) {
      log('AudioWorklet blocked, falling back to ScriptProcessor:', e);
      ({ Engine } = await import(chrome.runtime.getURL('engine.js')));
      holder.engine = newEngine(Engine);
      node = scriptProcessor(ctx, holder);
      mode = 'ScriptProcessor';
    }
    // Only now take over the element's sound: everything that can fail has succeeded.
    const src = ctx.createMediaElementSource(video);
    src.connect(node);
    node.connect(ctx.destination);
    current = { ctx, node, mode, holder, video, ahead, stats: { nMutes: 0, late: 0 } };
    if (node.port) {
      node.port.onmessage = (e) => {
        if (e.data.type === 'stats') current.stats = { nMutes: e.data.nMutes, late: e.data.late };
        else if (e.data.type === 'levels' && ahead) ahead.onLevels(e.data.pairs);
      };
    }
    if (ahead) {
      const q = pageQueue || [];
      pageQueue = null;
      for (const d of q) ahead.onPageMessage(d);
    } else pageQueue = null;
    // Autoplay rules may start the context suspended; any play or user input resumes it.
    const resume = () => { if (ctx.state !== 'running') ctx.resume(); };
    video.addEventListener('play', resume);
    document.addEventListener('click', resume, true);
    document.addEventListener('keydown', resume, true);
    resume();
    // The live engine needs to know about anything that breaks the audio stream or changes its time base.
    const reset = (newVideo) => {
      if (node.port) node.port.postMessage({ type: 'reset', ...eff(), newVideo, rate: video.playbackRate });
      else {
        const carry = newVideo ? null : holder.engine.carryState();
        const n = holder.engine.nMutes;
        holder.engine = newEngine(Engine, carry);
        holder.engine.nMutes = n;
        holder.engine.setRate(video.playbackRate);
      }
    };
    video.addEventListener('loadstart', () => reset(true));     // a new video in the same element
    video.addEventListener('seeked', () => reset(false));       // jumped: the old audio is gone
    video.addEventListener('play', () => reset(false));         // after a pause: drop the silence
    video.addEventListener('ratechange', () => {
      if (node.port) node.port.postMessage({ type: 'rate', rate: video.playbackRate });
      else holder.engine.setRate(video.playbackRate);
    });
    if (!node.port) holder.engine.setRate(video.playbackRate);
    log(`active (${mode}), operating point ${settings.fa} false alarms/h, attenuation ${settings.atten} dB`);
  }

  function apply() {
    if (!current) return;
    if (current.node.port) current.node.port.postMessage({ type: 'settings', ...eff() });
    else {
      const e = current.holder.engine;
      e.bypass = !eff().enabled;
      e.setOperatingPoint(model, settings.fa);
      e.setAttenuation(settings.atten);
    }
    if (current.ahead) current.ahead.setSettings(settings);
  }

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'sync') return;
    for (const [k, v] of Object.entries(changes)) if (k in DEFAULTS) settings[k] = v.newValue;
    apply();
  });

  chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
    if (msg.type === 'getStats') {
      const e = current?.holder.engine;
      const s = current && (e ? { nMutes: e.nMutes, late: e.late } : current.stats);
      reply({ channel: channel.id, watch: !!channel.videoId, allowed: allowed(), attached: !!current, mode: current?.mode, state: current?.ctx.state, ...(s || {}),
              ahead: current?.ahead ? current.ahead.status() : null });
    }
  });

  init().catch((e) => log('init failed:', e));
})();
