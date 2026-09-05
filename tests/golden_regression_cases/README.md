# Golden regression cases

These cases are evaluation-only and must never be added to any
training, fine-tuning, replay, synthetic generation, or calibration
dataset.

## Purpose

Permanent product/regression coverage for:

- Real held-out topomap (V2) and spectrogram (V3) vision-quality gates
- Routing phrases that previously misrouted to TEXT concept mode
  (`explain this image` vs `Explain motor imagery.`)

## Rules

1. **evaluation_only / training_allowed: false** on all real V2/V3 case metadata.
2. Do **not** copy assets from `assets/` into `data/` or any training pipeline.
3. Baseline VLM answers are **accepted production behavior snapshots**, not
   scientific ground truth. Open-ended topomap/spectrogram reading remains
   unreliable; product tests assert routing, provenance, isolation, and the
   experimental Limitations note — not scientific correctness.
4. Waveform / Vertex assets here are for regression/smoke convenience only;
   same evaluation-only rule applies.

## Layout

| Path | Role |
|------|------|
| `assets/v2_scalp_topomap_beta.png` | Real held-out V2 topomap figure |
| `assets/v3_spectrogram_tf.png` | Real held-out V3 spectrogram figure |
| `assets/Vertex_waves_EEG.png` | Waveform-style figure for routing/smoke |
| `cases/v2_topomap.json` | V2 metadata + production baseline output |
| `cases/v3_spectrogram.json` | V3 metadata + production baseline output |
| `test_golden_regression.py` | Automated product regressions |
