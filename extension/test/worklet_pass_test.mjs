// Testbench for the worklet's pass-through mode (videos from other channels): output must equal
// input sample for sample, with no delay, and nothing posted. Then, after pass-through is
// switched off, the live engine is back (delayed output).
//
// Usage: node extension/test/worklet_pass_test.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const model = JSON.parse(fs.readFileSync(path.join(here, '..', 'model.json'), 'utf8'));
let Proc = null;
globalThis.sampleRate = 48000;
globalThis.currentTime = 0;
globalThis.AudioWorkletProcessor = class { constructor() { this.port = { posted: [], postMessage(m) { this.posted.push(m); } }; } };
globalThis.registerProcessor = (name, cls) => { Proc = cls; };
await import('../worklet.js');

const p = new Proc({ processorOptions: { model, fa: '2', atten: 30, enabled: true, rate: 1 } });
const send = (m) => p.port.onmessage({ data: m });
let fail = 0;
const check = (name, ok, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  ' + info}`); if (!ok) fail++; };

function run(blocks, seed) {
  let maxErr = 0, firstNonZeroOut = -1, n = 0;
  for (let b = 0; b < blocks; b++) {
    const L = new Float32Array(128), R = new Float32Array(128);
    for (let i = 0; i < 128; i++) { L[i] = Math.sin((n + i + seed) * 0.01) * 0.5; R[i] = -L[i]; }
    const oL = new Float32Array(128), oR = new Float32Array(128);
    p.process([[L, R]], [[oL, oR]]);
    for (let i = 0; i < 128; i++) {
      maxErr = Math.max(maxErr, Math.abs(oL[i] - L[i]), Math.abs(oR[i] - R[i]));
      if (firstNonZeroOut < 0 && oL[i] !== 0) firstNonZeroOut = n + i;
    }
    n += 128;
    globalThis.currentTime += 128 / 48000;
  }
  return { maxErr, firstNonZeroOut };
}

send({ type: 'pass', on: true });
p.port.posted.length = 0;
const a = run(375, 1);                                    // 1 s
check('pass-through: output == input exactly', a.maxErr === 0, `max error ${a.maxErr}`);
check('pass-through: no delay', a.firstNonZeroOut === 0, `first output sample ${a.firstNonZeroOut}`);
check('pass-through: nothing posted (no stats, no levels)', p.port.posted.length === 0, `${p.port.posted.length} messages`);

send({ type: 'pass', on: false });
send({ type: 'reset', fa: '2', atten: 30, enabled: true, newVideo: true, rate: 1 });
const b = run(375, 1);
check('after pass-through: live engine delays the audio again', b.firstNonZeroOut > 0, `first output sample ${b.firstNonZeroOut}`);
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
