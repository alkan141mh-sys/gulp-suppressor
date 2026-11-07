"""Is the input representation the bottleneck? A CNN on a FINER time-frequency patch.

Every model so far saw the same input: 25 ms windows every 10 ms, 24 bands. Here the patch
around each anchor is recomputed from the audio with
  10 ms Hann windows every 5 ms (twice the time resolution, 2.5x shorter windows)
  40 mel bands 50 Hz - 8 kHz (vs 24)
over two spans: -30..+100 ms (as now) and -30..+200 ms (read-ahead makes the longer one free).

Model: 2-D CNN (time x frequency image): conv 5x5 (8 filters), max-pool over frequency by 2,
conv 3x3 (16 filters), max and mean over time, then one logistic unit with the 3 extras
(floor vs loud speech, anchor kind, time since onset). Same augmentation as cnn_detector.py
(shift +-2 frames, gain +-6 dB, tilt +-3 dB, 1 dB jitter), 3 seeds per fold, same CV.

Usage: .venv\\Scripts\\python tools\\cnn_fine.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

import gulpdata
import onset_classifier as oc
import reanchor as ra

FS = 16000
WIN, HOP, NFFT, NMEL = 160, 80, 512, 40
SHIFT = 2
torch.set_num_threads(6)


def mel_fb():
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(50), mel(8000), NMEL + 2))
    bins = np.fft.rfftfreq(NFFT, 1 / FS)
    fb = np.zeros((NMEL, len(bins)))
    for i in range(NMEL):
        a, c, b = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - a) / (c - a), (b - bins) / (b - c)), 0, None)
    return fb.astype(np.float32)


FB = mel_fb()
HANN = np.hanning(WIN).astype(np.float32)


def fine_patches(lec, f, D, span_ms):
    x, fs = gulpdata.read_wav(lec)
    y = resample_poly(x.astype(np.float32) / 32768, 1, 3).astype(np.float32)
    on, floor = oc.detect_onsets(f['total'])
    A, grp, kind = D['A'], D['grp'], D['kind']
    nfr = (30 + span_ms) // 5 + 2 * SHIFT
    s0 = np.round(A * 160 - (30 + 5 * SHIFT) * 16).astype(int)            # 16 samples per ms
    out = np.empty((len(A), NMEL, nfr), np.float32)
    for b in range(0, len(A), 256):
        st = s0[b:b + 256, None, None] + np.arange(nfr)[None, :, None] * HOP + np.arange(WIN)[None, None, :]
        fr = y[np.clip(st, 0, len(y) - 1)] * HANN
        P = np.abs(np.fft.rfft(fr, NFFT, axis=-1)) ** 2
        out[b:b + 256] = (10 * np.log10(P @ FB.T + 1e-10)).transpose(0, 2, 1)
    # Relative to the room floor; that floor is on the features' dB scale, which differs from this
    # one by a constant that the per-band standardisation removes.
    out -= floor[A][:, None, None]
    extra = np.column_stack([floor[A] - np.percentile(f['total'], 95), kind, (A - on[grp]) * 10]).astype(np.float32)
    return out, extra


class Net(nn.Module):
    def __init__(self, extra=3):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 5, padding=2)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(2 * 16 * (NMEL // 2) + extra, 1)

    def forward(self, x, e):
        z = F.relu(self.c1(x[:, None]))                    # (n, 8, mel, t)
        z = F.max_pool2d(z, (2, 1))
        z = F.relu(self.c2(z))                             # (n, 16, mel/2, t)
        z = torch.cat([z.amax(-1).flatten(1), z.mean(-1).flatten(1), e], 1)
        return self.fc(self.drop(z)).squeeze(1)


def augment(x, rng):
    n, nb, T = x.shape
    W = T - 2 * SHIFT
    off = rng.integers(0, 2 * SHIFT + 1, n)
    out = np.stack([x[i, :, o:o + W] for i, o in enumerate(off)])
    g = rng.uniform(-6, 6, (n, 1, 1)).astype(np.float32)
    tilt = rng.uniform(-3, 3, (n, 1, 1)).astype(np.float32) * np.linspace(-1, 1, nb, dtype=np.float32)[None, :, None]
    return out + g + tilt + rng.normal(0, 1, out.shape).astype(np.float32)


def center(x):
    return x[:, :, SHIFT:x.shape[2] - SHIFT]


def train(X, E, y, w, seed, epochs=12, pos_rep=30, neg_frac=0.35):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Xc = center(X)
    mu = Xc.mean(axis=(0, 2), keepdims=True); sd = Xc.std(axis=(0, 2), keepdims=True) + 1e-3
    emu, esd = E.mean(0), E.std(0) + 1e-3
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        ns = rng.choice(neg, int(len(neg) * neg_frac), replace=False)
        order = np.concatenate([ns, np.repeat(pos, pos_rep)])
        rng.shuffle(order)
        wneg = w[ns].sum()
        net.train()
        for b in range(0, len(order), 256):
            i = order[b:b + 256]
            xb = (augment(X[i], rng) - mu) / sd
            ww = np.where(y[i] == 1, wneg / (len(pos) * pos_rep), w[i]).astype(np.float32)
            logits = net(torch.from_numpy(xb.astype(np.float32)), torch.from_numpy((E[i] - emu) / esd))
            loss = (F.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[i].astype(np.float32)), reduction='none')
                    * torch.from_numpy(ww)).sum() / ww.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def score(Xt, Et):
        z = []
        with torch.no_grad():
            for b in range(0, len(Xt), 2048):
                xb = ((center(Xt[b:b + 2048]) - mu) / sd).astype(np.float32)
                z.append(net(torch.from_numpy(xb), torch.from_numpy((Et[b:b + 2048] - emu) / esd)).numpy())
        return np.concatenate(z).astype(float)
    return score


def cv(D, lectures, key, seeds=3):
    out = {}
    for fold in [lectures[i::oc.N_FOLDS] for i in range(oc.N_FOLDS)]:
        tr = [l for l in lectures if l not in fold]
        sel = {l: D[l]['y'] >= 0 for l in tr}
        X = np.concatenate([D[l][key][sel[l]] for l in tr]); E = np.concatenate([D[l]['E'][sel[l]] for l in tr])
        y = np.concatenate([D[l]['y'][sel[l]] for l in tr]); w = np.concatenate([D[l]['w'][sel[l]] for l in tr])
        acc = {l: 0 for l in fold}
        for s in range(seeds):
            score = train(X, E, y, w, seed=1000 * s + len(out))
            for l in fold:
                acc[l] = acc[l] + score(D[l][key], D[l]['E']) / seeds
        out.update(acc)
        print(f'  fold done ({len(out)}/{len(lectures)} lectures)', flush=True)
    return out


def main():
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {}
    for l in lectures:
        d = ra.lecture_data(feats[l], gulpdata.load_labels(l))
        d['total'] = feats[l]['total']
        d['F130'], d['E'] = fine_patches(l, feats[l], d, 100)
        d['F230'], _ = fine_patches(l, feats[l], d, 200)
        D[l] = d
    n = sum(len(d['gulps']) for d in D.values())
    print(f'{n} gulps, {hours:.1f} h; fine patches {D[lectures[0]]["F130"].shape[1:]} and {D[lectures[0]]["F230"].shape[1:]}\n', flush=True)
    base = ra.cv(D, lectures, False)
    rows = [('logistic, coarse patch (current)', base)]
    for name, key in [('2-D CNN, fine, -30..+100 ms', 'F130'), ('2-D CNN, fine, -30..+200 ms', 'F230')]:
        rows.append((name, cv(D, lectures, key)))
    rows.append(('average: current + fine +200 ms', {l: (base[l] + rows[-1][1][l]) / 2 for l in lectures}))
    print(f'\n{"model":<36}' + ''.join(f'{"@ " + str(t) + " FA/h":>12}' for t in oc.FA_TARGETS))
    for name, s in rows:
        print(f'{name:<36}' + ''.join(f'{a:>12.0%}' for a, _, _ in ra.evaluate(D, s, lectures, hours, oc.FA_TARGETS)))
    print(f'\n(one gulp = {100 / n:.1f} points; 95% CI at 70% about +-{196 * np.sqrt(.7 * .3 / n):.0f} points)')


if __name__ == '__main__':
    main()
