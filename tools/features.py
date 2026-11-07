"""Frame-level features for the offline candidate finder (floating point, not bit-true).

Frames: 25 ms Hann window, 10 ms hop, at 16 kHz. The lectures seem to have almost no
content above 8 kHz, so 16 kHz (Nyquist = 8 kHz) loses nothing for detection; the
high-frequency check below measures that per lecture instead of assuming it.
"""
import os

import numpy as np
from scipy.signal import resample_poly

import gulpdata

FS = 16000
WIN = 400        # 25 ms: long enough for 2+ pitch periods of an 80 Hz voice
HOP = 160        # 10 ms
NFFT = 1024      # >= 2*WIN, so the FFT-based autocorrelation is linear, not circular
F_LO, F_HI = 100.0, 8000.0
N_BANDS = 24
LAG_MIN, LAG_MAX = FS // 400, FS // 80   # pitch search 80-400 Hz
CACHE_VERSION = 2   # 2: bands summed directly in float64 (was a float32 cumsum difference)

_window = np.hanning(WIN).astype(np.float32)
# Autocorrelation of the window itself; dividing by it removes the taper's bias (Boersma 1993).
_win_ac = np.fft.irfft(np.abs(np.fft.rfft(_window, NFFT)) ** 2, NFFT)


def _band_edges():
    df = FS / NFFT
    e = np.round(np.geomspace(F_LO, F_HI, N_BANDS + 1) / df).astype(int)
    for i in range(1, len(e)):            # low bands would otherwise get zero bins
        e[i] = max(e[i], e[i - 1] + 1)
    return e


BAND_EDGES = _band_edges()


def high_freq_fraction(x48, fs):
    """Fraction of energy above 8 kHz in the original-rate audio."""
    n = 1 << 16
    acc = np.zeros(n // 2 + 1)
    for i in range(0, len(x48) - n + 1, n):
        acc += np.abs(np.fft.rfft(x48[i:i + n])) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    return float(acc[f > 8000].sum() / max(acc.sum(), 1e-30))


def frame_stats(frames):
    """Bands, total level, harmonicity and flatness of raw (unwindowed) frames, shape (n, WIN).

    Shared by the batch features below and the streaming engine (live.py), so the live
    detector sees exactly the numbers the model was trained on.
    """
    lo, hi = BAND_EDGES[0], BAND_EDGES[-1]
    fr = frames * _window
    P = (np.abs(np.fft.rfft(fr, NFFT, axis=1)) ** 2).astype(np.float64)
    # Each band is summed directly. Taking differences of a cumulative sum (the first version)
    # cancels catastrophically in float32: a band 60 dB below the frame's strongest bins came
    # out up to 0.4 dB wrong. A direct sum stays accurate, also later in fixed point.
    bands = 10 * np.log10(np.add.reduceat(P[:, :hi], BAND_EDGES[:-1], axis=1) + 1e-12)
    Pin = P[:, lo:hi]
    total = 10 * np.log10(Pin.sum(1) + 1e-12)
    flat = np.exp(np.log(Pin + 1e-12).mean(1)) / (Pin.mean(1) + 1e-12)
    ac = np.fft.irfft(P, NFFT, axis=1)
    r = ac[:, LAG_MIN:LAG_MAX + 1] / _win_ac[LAG_MIN:LAG_MAX + 1] / (ac[:, :1] / _win_ac[0] + 1e-12)
    return bands, total, np.clip(r.max(1), 0, 1), flat


def compute(lec):
    x16, fs = gulpdata.read_wav(lec)
    x = x16.astype(np.float32) / 32768.0          # Q15 input -> [-1, 1)
    hf = high_freq_fraction(x, fs)
    x = resample_poly(x, FS, fs).astype(np.float32)  # includes the anti-alias low-pass

    T = 1 + (len(x) - WIN) // HOP
    bands = np.empty((T, N_BANDS), np.float32)
    total = np.empty(T, np.float32)
    harm = np.empty(T, np.float32)
    flat = np.empty(T, np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP][:T]
    for a in range(0, T, 20000):                  # chunks keep memory around 100 MB
        b, t, h, fl = frame_stats(frames[a:a + 20000])
        n = len(t)
        bands[a:a + n], total[a:a + n], harm[a:a + n], flat[a:a + n] = b, t, h, fl
    return {'bands': bands, 'total': total, 'harm': harm, 'flat': flat,
            'hf_fraction': np.float32(hf), 'version': np.int32(CACHE_VERSION)}


def lecture_features(lec):
    """Cached per-lecture features (data/cache/<lec>.npz)."""
    path = os.path.join(gulpdata.CACHE, f'{lec}.npz')
    if os.path.exists(path):
        with np.load(path) as z:
            if int(z['version']) == CACHE_VERSION:
                return {k: z[k] for k in z.files}
    f = compute(lec)
    os.makedirs(gulpdata.CACHE, exist_ok=True)
    np.savez(path, **f)
    return f


def frame_time(i):
    """Start time in seconds of frame i."""
    return i * HOP / FS
