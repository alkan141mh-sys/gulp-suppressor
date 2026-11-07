"""Write extension/model.json: the trained detector plus every constant the JavaScript engine
needs (resampler taps, analysis window, band edges, window autocorrelation).

Exporting the constants, instead of recomputing them in JavaScript, keeps the two engines on
exactly the same numbers. Run after train_final.py.

Usage: .venv\\Scripts\\python tools\\export_extension.py
"""
import json
import os

import features
import gulpdata
import live

OUT = os.path.join(gulpdata.ROOT, 'extension', 'model.json')


def main():
    with open(live.MODEL, encoding='utf-8') as f:
        model = json.load(f)
    model['dsp'] = {
        'FS_IN': live.FS_IN, 'DOWN': live.DOWN, 'HALF': live.HALF, 'taps': live.H.tolist(),
        'WIN': features.WIN, 'HOP': features.HOP, 'NFFT': features.NFFT,
        'window': features._window.astype(float).tolist(),
        'band_edges': features.BAND_EDGES.tolist(),
        'lag_min': int(features.LAG_MIN), 'lag_max': int(features.LAG_MAX),
        'win_ac': features._win_ac[:features.LAG_MAX + 1].tolist(),
        'DELAY_MS': live.DELAY_MS, 'RAMP_MS': live.RAMP_MS, 'MUTE_END_QUIET': live.MUTE_END_QUIET,
        'MUTE_MAX_S': live.MUTE_MAX_S, 'P95_FRAMES': live.P95_FRAMES,
        'SPEECH_HARM': live.SPEECH_HARM, 'SPEECH_DB': live.SPEECH_DB, 'SPEECH_FRAMES': live.SPEECH_FRAMES,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(model, f)
    print(f'saved {os.path.relpath(OUT, gulpdata.ROOT)} ({os.path.getsize(OUT) / 1024:.0f} KiB)')


if __name__ == '__main__':
    main()
