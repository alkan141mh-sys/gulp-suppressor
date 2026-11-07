// Seek test: for every labelled gulp, "seek" to SEEK_BEFORE s before it, as the content script
// does on 'seeked' (a new engine, optionally carrying floor / speech level from the old one),
// play 3 s and check whether the gulp gets muted.
//   node extension/test/seek_test.mjs <in.wav> <gulps.json>   (gulps.json: [[start, end], ...])
import { readFileSync } from 'node:fs';
import { Engine } from '../engine.js';

const [, , wavPath, gulpsPath] = process.argv;
const SEEK_BEFORE = 1.0;
const model = JSON.parse(readFileSync(new URL('../model.json', import.meta.url)));
const gulps = JSON.parse(readFileSync(gulpsPath));
const buf = readFileSync(wavPath);
let pos = 12, data = null;
while (pos + 8 <= buf.length) {
  const id = buf.toString('ascii', pos, pos + 4), size = buf.readUInt32LE(pos + 4);
  if (id === 'data') { data = buf.subarray(pos + 8, pos + 8 + size); break; }
  pos += 8 + size + (size & 1);
}
const x = new Float32Array(data.length / 2);
for (let i = 0; i < x.length; i++) x[i] = data.readInt16LE(i * 2) / 32768;

function play(eng, from, seconds) {
  const B = 128, outs = [new Float32Array(B)];
  for (let a = from; a < from + seconds * 48000; a += B) eng.process([x.subarray(a, a + B)], outs);
}

// Warm one engine on the first minute, as if the viewer had been watching for a while.
const warm = new Engine(model, '15', 30, 1);
play(warm, 0, 60);
const carry = warm.carryState();
for (const [name, c] of [['carry floor and speech level', carry], ['start from scratch', null]]) {
  let hit = 0;
  for (const [s, e] of gulps) {
    const from = Math.max(0, Math.round((s - SEEK_BEFORE) * 48000));
    const eng = new Engine(model, '15', 30, 1, c);
    play(eng, from, SEEK_BEFORE + 2);
    const off = from / 48000;                       // engine time 0 = media time 'off'
    if (eng.muteLog.some((m) => m.start + off < e && (m.end === null ? Infinity : m.end + off) > s)) hit++;
  }
  console.log(`seek to ${SEEK_BEFORE} s before each gulp, ${name}: ${hit}/${gulps.length} gulps muted`);
}
