# tools/

Python side of the project. Every script starts with a docstring that says what it measures, why, and how to run it. Run them from the repository root, for example `.venv\Scripts\python tools\train_final.py`.

## Core (the shipped detector)

| Script | What it does |
|---|---|
| `gulpdata.py` | Shared access to lecture audio, label files and review decisions |
| `features.py` | Frame features: 24 log bands, harmonicity, flux (25 ms / 10 ms, 16 kHz) |
| `onset_classifier.py` | Stage 1 onset detector + stage 2 classifier on the 100 ms patch |
| `reanchor.py` | Secondary anchors inside a sound; `merged_gulps` counting |
| `breath_rule.py` | Gated breath-vs-gulp rule (`sound_feats`) |
| `train_final.py` | Final training, vetoes, audit-corrected thresholds → `models/detector.json` |
| `export_extension.py` | `models/detector.json` + constants → `extension/model.json` |
| `live.py` | Real-time engine in Python (sounddevice) and offline mode |
| `plot_final_roc.py` | README figure |

## Testbenches (Python ↔ JavaScript)

| Script | What it checks |
|---|---|
| `check_live.py` | Streaming engine computes what the batch analysis computes |
| `check_extension.py` | `extension/engine.js` decides exactly like `live.py` |
| `check_breath_js.py` | Breath/click veto in JS matches `train_final.shape_veto` |

## Evaluation

These scripts measure detection quality: ROC curves against lookahead, mute edges, the false-alarm rate corrected with the audits, and the learning curve.

`latency_roc.py`, `output_roc.py`, `plot_*.py`, `eval_mutes.py`, `eval_mute_edges.py`, `eval_mute_variants.py`, `corrected_roc.py`, `learning_curve.py`, `miss_analysis.py`, `relabel_effect.py`, `build_final_check.py`

## Labels and audits

- **Review tool:** `review_server.py` + `review.html`, a local labelling and review UI at http://127.0.0.1:8765.
- **Review queues:** `build_queues.py`, `build_confirmed_queue.py`, `find_candidates.py`, `export_*.py`.
- **Blind audits:** `build_audit.py`, `build_audit2.py`, `audit_result.py`, `audit_result2.py`, `audit_inspect.py`.
- **Label quality:** `label_quality.py`, `recheck_agreement.py` (Cohen's kappa), `pu_correction.py`.
- **Confirmed false alarms and annotator cues:** `build_fa_known.py`, `build_fa_hard.py`, `cue_check.py`, `click_before_speech.py`.

## Experiments that did not ship

Each of these is kept with its result in its docstring:

- `onset_models.py` compares stage 2 classifiers.
- `cnn_detector.py` is a small CNN with augmentation.
- `cnn_fine.py` is a CNN on a finer time-frequency patch.
- `whole_sound.py` classifies the whole sound.
- `click_shape.py` showed that the "clicks" were the mouse, not the mouth.
