"""Small CNN on the onset patch, with data augmentation, against the current logistic model.

Same units, labels and lecture-grouped 6-fold cross-validation as reanchor.py; the CNN sees the
same patch (frames -3..+7 around each anchor: 24 bands relative to the floor, harmonicity,
spectral flux) plus the same three extras (floor vs loud-speech level, anchor kind, time since
onset). Only the model changes:

  logistic   one weight per patch cell (289), as now
  CNN        two 1-D convolutions over time (the 26 per-frame values are the input channels,
             kernel 3 frames, 16 filters each, ReLU), then max- and mean-pooling over time, so a
             pattern counts wherever it falls in the window, then one logistic unit
             (~2,200 weights)

Augmentation (training only, on the features, applied to gulps and other sounds alike):
  time shift   the window is cut at a random offset of up to +-2 frames (20 ms)
  gain         all bands +-6 dB (gulps are 7-21 dB above the floor)
  tilt         a linear spectral slope of up to +-3 dB across the bands (microphone colour)
  jitter       1 dB Gaussian noise on every band value

Each setting is trained with 3 random seeds per fold and the log-odds are averaged.

Usage: .venv\\Scripts\\python tools\\cnn_detector.py [--window 11] [--seeds 3] [--fractions]
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import features
import gulpdata
import onset_classifier as oc
import reanchor as ra

SHIFT = 2
torch.set_num_threads(6)


def raw_patches(f, D, n_frames):
    """(anchors, 26, n_frames + 2*SHIFT) patches and (anchors, 3) extras."""
    total = f['total']
    T = len(total)
    on, floor = oc.detect_onsets(total)
    flux = np.concatenate([[0], np.maximum(np.diff(f['bands'], axis=0), 0).sum(axis=1)])
    per_frame = np.column_stack([f['bands'], f['harm'], flux]).astype(np.float32)
    A, grp, kind = D['A'], D['grp'], D['kind']
    idx = np.clip(A[:, None] + np.arange(-oc.PRE - SHIFT, n_frames - oc.PRE + SHIFT)[None, :], 0, T - 1)
    P = per_frame[idx]                                            # (n, frames, 26)
    P[:, :, :features.N_BANDS] -= floor[A][:, None, None]
    extra = np.column_stack([floor[A] - np.percentile(total, 95), kind, (A - on[grp]) * 10]).astype(np.float32)
    return np.ascontiguousarray(P.transpose(0, 2, 1)), extra


class Net(nn.Module):
    def __init__(self, cin=26, h=16, extra=3):
        super().__init__()
        self.c1 = nn.Conv1d(cin, h, 3, padding=1)
        self.c2 = nn.Conv1d(h, h, 3, padding=1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(2 * h + extra, 1)

    def forward(self, x, e):
        z = F.relu(self.c2(F.relu(self.c1(x))))
        z = torch.cat([z.amax(-1), z.mean(-1), e], 1)
        return self.fc(self.drop(z)).squeeze(1)


def augment(x, rng):
    n = x.shape[0]
    nb = features.N_BANDS
    off = rng.integers(0, 2 * SHIFT + 1, n)
    W = x.shape[2] - 2 * SHIFT
    out = np.stack([x[i, :, o:o + W] for i, o in enumerate(off)])
    g = rng.uniform(-6, 6, (n, 1, 1)).astype(np.float32)
    tilt = rng.uniform(-3, 3, (n, 1, 1)).astype(np.float32) * np.linspace(-1, 1, nb, dtype=np.float32)[None, :, None]
    out[:, :nb] += g + tilt + rng.normal(0, 1, (n, nb, W)).astype(np.float32)
    return out


def center(x):
    return x[:, :, SHIFT:x.shape[2] - SHIFT]


def train(X, E, y, w, aug, seed, epochs=25, pos_rep=30):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Xc = center(X)
    mu = Xc.mean(axis=(0, 2), keepdims=True); sd = Xc.std(axis=(0, 2), keepdims=True) + 1e-3
    emu, esd = E.mean(0), E.std(0) + 1e-3
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    rep = pos_rep if aug else 1
    wneg = w[neg].sum()
    for ep in range(epochs):
        order = np.concatenate([neg, np.repeat(pos, rep)])
        rng.shuffle(order)
        net.train()
        for b in range(0, len(order), 256):
            i = order[b:b + 256]
            xb = augment(X[i], rng) if aug else Xc[i]
            ww = np.where(y[i] == 1, wneg / (len(pos) * rep), w[i]).astype(np.float32)
            logits = net(torch.from_numpy((xb - mu) / sd), torch.from_numpy((E[i] - emu) / esd))
            loss = (F.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[i].astype(np.float32)), reduction='none')
                    * torch.from_numpy(ww)).sum() / ww.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def score(Xt, Et):
        with torch.no_grad():
            return net(torch.from_numpy((center(Xt) - mu) / sd), torch.from_numpy((Et - emu) / esd)).numpy().astype(float)
    return score


def cv(D, lectures, aug, seeds, keep=None):
    """Cross-validated CNN log-odds per lecture. keep: optional {lecture: y used for training}."""
    out = {}
    for fold in [lectures[i::oc.N_FOLDS] for i in range(oc.N_FOLDS)]:
        tr = [l for l in lectures if l not in fold]
        ys = {l: (keep[l] if keep else D[l]['y']) for l in tr}
        sel = {l: ys[l] >= 0 for l in tr}
        X = np.concatenate([D[l]['P'][sel[l]] for l in tr]); E = np.concatenate([D[l]['E'][sel[l]] for l in tr])
        y = np.concatenate([ys[l][sel[l]] for l in tr]); w = np.concatenate([D[l]['w'][sel[l]] for l in tr])
        acc = {l: 0 for l in fold}
        for s in range(seeds):
            score = train(X, E, y, w, aug, seed=1000 * s + len(out))
            for l in fold:
                acc[l] = acc[l] + score(D[l]['P'], D[l]['E']) / seeds
        out.update(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=11, help='patch frames (11 = -30..+80 ms, as now)')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--fractions', action='store_true', help='also a learning curve for the augmented CNN')
    args = ap.parse_args()
    lectures, feats, _ = oc.load_all()
    hours = sum(len(feats[l]['total']) for l in lectures) * oc.HOP_S / 3600
    D = {}
    for l in lectures:
        d = ra.lecture_data(feats[l], gulpdata.load_labels(l))
        d['total'] = feats[l]['total']
        d['P'], d['E'] = raw_patches(feats[l], d, args.window)
        D[l] = d
    n = sum(len(d['gulps']) for d in D.values())
    print(f'{n} gulps, {hours:.1f} h, window {args.window} frames, {args.seeds} seeds per fold\n')
    print(f'{"model":<34}' + ''.join(f'{"@ " + str(t) + " FA/h":>12}' for t in oc.FA_TARGETS))
    base = ra.cv(D, lectures, False)
    show = lambda name, s: print(f'{name:<34}' + ''.join(f'{a:>12.0%}' for a, _, _ in ra.evaluate(D, s, lectures, hours, oc.FA_TARGETS)), flush=True)
    show('logistic (current)', base)
    plain = cv(D, lectures, False, args.seeds)
    show('CNN, no augmentation', plain)
    augd = cv(D, lectures, True, args.seeds)
    show('CNN, augmented', augd)
    show('average: logistic + CNN augmented', {l: (base[l] + augd[l]) / 2 for l in lectures})
    if args.fractions:
        print('\nlearning curve, CNN augmented (one random subset each):')
        for frac in (0.25, 0.5, 0.75):
            rng = np.random.default_rng(7)
            keep = {}
            for l in lectures:
                y = D[l]['y'].copy()
                for g in D[l]['gulps']:
                    if rng.random() >= frac:
                        a = g['anchors']; y[a[y[a] == 1]] = -1
                keep[l] = y
            show(f'  {frac:.0%} of the gulps', cv(D, lectures, True, args.seeds, keep))
    print(f'\n(one gulp = {100 / n:.1f} points; 95% CI at 70% about +-{196 * np.sqrt(.7 * .3 / n):.0f} points)')


if __name__ == '__main__':
    main()
