// Run the extension engine over a WAV file in 128-sample blocks, exactly as an AudioWorklet
// would call it, and write every decision and mute to JSON. Used by tools/check_extension.py
// and tools/eval_mutes.py.
//   node extension/test/run_lecture.mjs <in.wav> <events.json> [--rate 1.5] [--old-mutes]
// --rate tells the engine the playback speed (as video.playbackRate would);
// --old-mutes restores the first version's mute rule (no speech check, 0.8 s cap).
import { readFileSync, writeFileSync } from 'node:fs';
import { Engine } from '../engine.js';

const args = process.argv.slice(2);
const [wavPath, outPath] = args;
const opt = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; };
const model = JSON.parse(readFileSync(new URL('../model.json', import.meta.url)));
if (args.includes('--old-mutes')) { model.dsp.SPEECH_HARM = 99; model.dsp.MUTE_MAX_S = 0.8; }
const buf = readFileSync(wavPath);

// Minimal WAV reader: 16-bit PCM, find the 'fmt ' and 'data' chunks.
let pos = 12, ch = 1, fs = 0, data = null;
while (pos + 8 <= buf.length) {
  const id = buf.toString('ascii', pos, pos + 4), size = buf.readUInt32LE(pos + 4);
  if (id === 'fmt ') { ch = buf.readUInt16LE(pos + 10); fs = buf.readUInt32LE(pos + 12); }
  if (id === 'data') { data = buf.subarray(pos + 8, pos + 8 + size); break; }
  pos += 8 + size + (size & 1);
}
if (fs !== 48000) throw new Error(`expected 48 kHz, got ${fs}`);
const n = data.length / 2 / ch;
const chans = Array.from({ length: ch }, () => new Float32Array(n));
for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) chans[c][i] = data.readInt16LE((i * ch + c) * 2) / 32768;

const eng = new Engine(model, opt('--fa') || '15', 30, ch);
if (opt('--rate')) eng.setRate(Number(opt('--rate')));
if (args.includes('--from-onset')) eng.fromOnset = true;       // read-ahead mute options
if (opt('--preroll')) eng.preRoll = Number(opt('--preroll')) / 1000;
if (opt('--veto-frames')) eng.vetoFrames = Number(opt('--veto-frames'));
if (opt('--veto-db')) eng.vetoDb = Number(opt('--veto-db'));
if (args.includes('--breath')) eng.breathRule = true;
if (opt('--ds-ahead')) eng.dsAhead = Number(opt('--ds-ahead'));
const B = 128, outs = Array.from({ length: ch }, () => new Float32Array(B));
const t = performance.now();
for (let a = 0; a + B <= n + eng.delay; a += B) {
  const ins = chans.map(x => { const b = new Float32Array(B); b.set(x.subarray(a, Math.min(a + B, n))); return b; });
  eng.process(ins, outs);
}
const sec = (performance.now() - t) / 1000;
const mutes = eng.muteLog.map((m) => [m.start, m.end === null ? m.start + model.dsp.MUTE_MAX_S / eng.rate : m.end]);
writeFileSync(outPath, JSON.stringify({ events: eng.events, mutes, late: eng.late, frames: eng.k, seconds: sec, audio_s: n / fs }));
console.log(`${(n / fs / 60).toFixed(1)} min in ${sec.toFixed(1)} s (${(n / fs / sec).toFixed(0)}x real time), ` +
            `${eng.events.length} decisions, ${eng.nMutes} mutes, late ${eng.late}`);
