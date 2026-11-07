// Analyzer test (Node): decoded Opus audio fed as 20 ms blocks stamped with media time, as the
// browser's AudioDecoder delivers it, must give the same mutes as the live engine run over the
// original WAV (run_lecture.mjs), apart from what Opus coding changes. A second run feeds two
// separate stretches, as after a seek, and checks that the second one carries floor and level.
//   node extension/test/ahead_test.mjs <decoded_f32.wav> <live_mutes.json>
import { readFileSync } from 'node:fs';
import { Analyzer } from '../ahead_core.js';

const [wavPath, livePath, FA = '15'] = process.argv.slice(2);
const model = JSON.parse(readFileSync(new URL('../model.json', import.meta.url)));
const live = JSON.parse(readFileSync(livePath)).mutes;

function readF32(path) {
  const b = readFileSync(path);
  let pos = 12, data = null, fmt = 0, ch = 1;
  while (pos + 8 <= b.length) {
    const id = b.toString('ascii', pos, pos + 4), size = b.readUInt32LE(pos + 4);
    if (id === 'fmt ') { fmt = b.readUInt16LE(pos + 8); ch = b.readUInt16LE(pos + 10); }
    if (id === 'data') { data = b.subarray(pos + 8, pos + 8 + size); break; }
    pos += 8 + size + (size & 1);
  }
  if (fmt !== 3 || ch !== 1) throw new Error('expected mono float32 WAV');
  return new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}

const x = readF32(wavPath);
const B = 960;

function feed(an, a, b) {
  for (let i = Math.floor(a * 48000 / B) * B; i + B <= Math.min(x.length, b * 48000); i += B) an.add(i / 48000, x.slice(i, i + B));
  while (an.step(1e9));
}

function compare(name, got, ref, range = [0, Infinity]) {
  const r = ref.filter(m => m[0] >= range[0] && m[0] < range[1]);
  const g = got.filter(m => m[0] >= range[0] && m[0] < range[1]);
  let matched = 0; const ds = [], de = [];
  for (const m of r) {
    const h = g.find(q => Math.abs(q[0] - m[0]) < 0.05);
    if (h) { matched++; ds.push((h[0] - m[0]) * 1000); de.push((h[1] - m[1]) * 1000); }
  }
  const med = a => a.length ? [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)] : NaN;
  const res = { name, live: r.length, ahead: g.length, matched, start_diff_ms_median: med(ds),
                start_diff_ms_max: Math.max(0, ...ds.map(Math.abs)), end_diff_ms_median: med(de) };
  console.log(JSON.stringify(res));
  return res;
}

// 1) one stretch, the whole excerpt
const an = new Analyzer(model, FA);
const t = performance.now();
feed(an, 0, Infinity);
const sec = (performance.now() - t) / 1000;
console.log(`analysed ${(x.length / 48000).toFixed(0)} s in ${sec.toFixed(1)} s (${(x.length / 48000 / sec).toFixed(0)}x real time), segments ${an.segs.length}`);
const r1 = compare('whole', an.mutes(), live);

// 2) as after a seek: 0-100 s, then 200 s onwards. The live run is continuous, so compare only
//    mutes after the second stretch has warmed up (and its floor tracker has a full second).
const an2 = new Analyzer(model, FA);
feed(an2, 0, 100);
feed(an2, 200, Infinity);
console.log(`seek run: segments ${an2.segs.length} at ${an2.segs.map(s => s.t0.toFixed(2)).join(', ')}`);
const r2 = compare('after seek', an2.mutes(), live, [201.5, Infinity]);

// 3) changing the operating point re-analyses the held audio
an2.evict(250);
an2.setFa(FA === '5' ? '20' : '5');
while (an2.step(1e9));
const ref5 = new Analyzer(model, FA === '5' ? '20' : '5'); feed(ref5, 0, Infinity);
compare('fa 5 after re-analysis vs fresh fa 5', an2.mutes(), ref5.mutes(), [252, Infinity]);

const ok = r1.matched >= r1.live - 1 && r2.matched >= r2.live - 1;
console.log(ok ? 'PASS' : 'FAIL');

