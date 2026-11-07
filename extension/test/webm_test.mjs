// Demuxer test: feed a WebM/Opus file to WebmDemuxer in random-sized pieces (as appendBuffer
// calls might arrive) and compare every frame's time and size with ffprobe's packet list.
//   node extension/test/webm_test.mjs results/ahead/005_excerpt.webm results/ahead/packets.csv
import { readFileSync } from 'node:fs';
import { WebmDemuxer } from '../webm.js';

const [webmPath, csvPath] = process.argv.slice(2);
const file = new Uint8Array(readFileSync(webmPath));
const ref = readFileSync(csvPath, 'utf8').trim().split(/\r?\n/).map(l => l.split(',')).map(([t, s]) => [Number(t), Number(s)]);

let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32;

function run(pieceMax) {
  const dm = new WebmDemuxer();
  const frames = [];
  for (let o = 0; o < file.length;) {
    const n = Math.max(1, Math.floor(rnd() * pieceMax));
    frames.push(...dm.push(file.subarray(o, o + n)));
    o += n;
  }
  const tr = dm.audioTrack();
  const got = frames.filter(f => f.track === tr.number);
  let bad = 0;
  for (let i = 0; i < Math.max(got.length, ref.length); i++) {
    const g = got[i], r = ref[i];
    // ffprobe reports block time minus the Opus CodecDelay (6.5 ms here), on a 1 ms time base.
    if (!g || !r || Math.abs(g.time - tr.delay - r[0]) > 0.001 + 1e-9 || g.data.length !== r[1]) {
      if (bad < 3) console.log('mismatch at', i, g && [g.time, g.data.length], r);
      bad++;
    }
  }
  return { pieceMax, frames: got.length, expected: ref.length, mismatches: bad, errors: dm.errors,
           laced: dm.lacedSkipped, codec: tr.codec, rate: tr.rate, channels: tr.channels,
           codecDelay_ms: tr.delay * 1000, opusHead: tr.private && String.fromCharCode(...tr.private.subarray(0, 8)) };
}

let ok = true;
for (const pm of [1e9, 65536, 4096, 7, 1]) {
  if (pm === 1 && file.length > 2e7) continue;
  const r = run(pm);
  console.log(JSON.stringify(r));
  ok &&= r.mismatches === 0 && r.errors === 0 && r.frames === r.expected;
}
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
