// Testbench for page.js's channel gate: copies of YouTube's audio appends are posted only for
// Buders videos. Runs page.js in a VM against a mock MediaSource / SourceBuffer / document and
// checks what reaches postMessage in each navigation scenario.
//
// Usage: node extension/test/page_gate_test.mjs
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(here, '..', 'page.js'), 'utf8');
const BUDERS = 'UClKBd-4xFVodFmK3bZfJAAQ', OTHER = 'UCsXVk37bltHxD1rDPwtNM8Q';

function makePage() {
  const posted = [];
  const listeners = {};
  let copies = 0;
  class SourceBuffer { constructor() { this.timestampOffset = 0; } appendBuffer() {} abort() {} }
  class MediaSource {
    static isTypeSupported() { return true; }
    addSourceBuffer(type) { return new SourceBuffer(type); }
  }
  const page = {
    location: { pathname: '/', search: '', origin: 'https://www.youtube.com' },
    player: null,                                   // videoDetails the mock player reports
  };
  const sandbox = {
    MediaSource, SourceBuffer,
    URL: { createObjectURL: () => 'blob:x' },
    location: page.location,
    URLSearchParams,
    setTimeout: () => 0, clearTimeout: () => {},
    document: {
      addEventListener: (ev, fn) => { (listeners[ev] ||= []).push(fn); },
      getElementById: () => ({ getPlayerResponse: () => page.player && { videoDetails: page.player } }),
    },
    Uint8Array, ArrayBuffer,
  };
  sandbox.window = sandbox;
  sandbox.window.addEventListener = (ev, fn) => { (listeners[ev] ||= []).push(fn); };
  sandbox.window.postMessage = (msg) => { posted.push(msg); };
  // count copies: page.js copies with Uint8Array.prototype.slice
  const slice = Uint8Array.prototype.slice;
  sandbox.Uint8Array = class extends Uint8Array { slice(...a) { copies++; return slice.apply(this, a); } };
  vm.createContext(sandbox);
  vm.runInContext(SRC, sandbox);
  const fire = (ev) => (listeners[ev] || []).forEach(f => f());
  return {
    posted, fire, copies: () => copies, MS: sandbox.MediaSource,
    goto(pathname, v, channelId) {
      fire('yt-navigate-start');
      page.location.pathname = pathname;
      page.location.search = v ? `?v=${v}` : '';
      page.player = v ? { videoId: v, channelId } : null;
      fire('yt-navigate-finish');
    },
  };
}
const bytes = (n) => new ArrayBuffer(n);
const kinds = (p) => p.filter(m => m.kind !== 'channel').map(m => m.kind + (m.bytes ? ':' + m.bytes.byteLength : ''));

let fail = 0;
function check(name, cond, info = '') {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : '  ' + info}`);
  if (!cond) fail++;
}

// 1. Home page hover preview: nothing posted, nothing copied beyond the header.
{
  const p = makePage();
  p.goto('/', null);
  const sb = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  for (let i = 0; i < 50; i++) sb.appendBuffer(bytes(30000));
  check('home page: no audio posted', kinds(p.posted).length === 0, JSON.stringify(kinds(p.posted)));
  check('home page: only the header copied', p.copies() === 1, `copies ${p.copies()}`);
}
// 2. Other channel: nothing posted, nothing copied beyond the header.
{
  const p = makePage();
  p.goto('/watch', 'aaa', OTHER);
  const sb = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  for (let i = 0; i < 200; i++) sb.appendBuffer(bytes(12000));
  check('other channel: no audio posted', kinds(p.posted).length === 0, JSON.stringify(kinds(p.posted)));
  check('other channel: only the header copied', p.copies() === 1, `copies ${p.copies()}`);
}
// 3. Buders, channel known first: everything posted in order.
{
  const p = makePage();
  p.goto('/watch', 'bbb', BUDERS);
  const sb = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  sb.appendBuffer(bytes(500)); sb.appendBuffer(bytes(1000)); sb.abort(); sb.appendBuffer(bytes(2000));
  check('Buders: sb + appends + abort posted in order',
        JSON.stringify(kinds(p.posted)) === JSON.stringify(['sb', 'append:500', 'append:1000', 'abort', 'append:2000']),
        JSON.stringify(kinds(p.posted)));
}
// 4. Channel not yet known when the player starts buffering: held, then flushed for Buders.
{
  const p = makePage();
  p.fire('yt-navigate-start');                         // unknown
  const sb = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  sb.appendBuffer(bytes(500)); sb.appendBuffer(bytes(1000));
  check('unknown channel: nothing posted yet', kinds(p.posted).length === 0, JSON.stringify(kinds(p.posted)));
  p.goto('/watch', 'ccc', BUDERS);
  sb.appendBuffer(bytes(2000));
  check('unknown -> Buders: held data flushed, then live',
        JSON.stringify(kinds(p.posted)) === JSON.stringify(['sb', 'append:500', 'append:1000', 'append:2000']),
        JSON.stringify(kinds(p.posted)));
  const iCh = p.posted.findIndex(m => m.kind === 'channel' && m.id === BUDERS), iSb = p.posted.findIndex(m => m.kind === 'sb');
  check('channel message precedes the flushed data', iCh >= 0 && iCh < iSb, `channel ${iCh}, sb ${iSb}`);
}
// 5. Unknown, then another channel: held data dropped, nothing posted.
{
  const p = makePage();
  p.fire('yt-navigate-start');
  const sb = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  sb.appendBuffer(bytes(500)); sb.appendBuffer(bytes(1000));
  p.goto('/watch', 'ddd', OTHER);
  sb.appendBuffer(bytes(2000));
  check('unknown -> other: nothing posted', kinds(p.posted).length === 0, JSON.stringify(kinds(p.posted)));
}
// 6. Buders -> other (new SourceBuffer): the other video's audio is not posted.
{
  const p = makePage();
  p.goto('/watch', 'eee', BUDERS);
  const a = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  a.appendBuffer(bytes(500));
  p.goto('/watch', 'fff', OTHER);
  const b = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  b.appendBuffer(bytes(700)); b.appendBuffer(bytes(800)); a.appendBuffer(bytes(900));
  check('Buders -> other: only the Buders audio posted',
        JSON.stringify(kinds(p.posted)) === JSON.stringify(['sb', 'append:500']), JSON.stringify(kinds(p.posted)));
}
// 7. Other -> Buders reusing the same SourceBuffer: sb + header re-sent, then live appends.
{
  const p = makePage();
  p.goto('/watch', 'ggg', OTHER);
  const sb = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  sb.appendBuffer(bytes(400)); sb.appendBuffer(bytes(5000));
  p.goto('/watch', 'hhh', BUDERS);
  sb.appendBuffer(bytes(6000));
  check('other -> Buders, reused buffer: sb, header, then live',
        JSON.stringify(kinds(p.posted)) === JSON.stringify(['sb', 'append:400', 'append:6000']), JSON.stringify(kinds(p.posted)));
}
// 8. Waiting too long: the hold is bounded and keeps sb + header.
{
  const p = makePage();
  p.fire('yt-navigate-start');
  const sb = new p.MS().addSourceBuffer('audio/webm; codecs="opus"');
  sb.appendBuffer(bytes(300));
  for (let i = 0; i < 400; i++) sb.appendBuffer(bytes(100000));     // 40 MB
  p.goto('/watch', 'iii', BUDERS);
  const k = kinds(p.posted), audio = p.posted.filter(m => m.kind === 'append').reduce((s, m) => s + m.bytes.byteLength, 0);
  check('bounded hold keeps sb and header', k[0] === 'sb' && k[1] === 'append:300', JSON.stringify(k.slice(0, 3)));
  check('bounded hold is at most 16 MB', audio <= (16 << 20) + 300, `${audio} B`);
}

console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
