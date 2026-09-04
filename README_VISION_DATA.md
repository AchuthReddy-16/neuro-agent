# EEGMMIDB multimodal vision dataset

## Purpose and provenance

This dataset adds deterministic EEG-derived research plots to the existing PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0 pipeline. It contains no scraped or decorative imagery. Every PNG is generated from the processed EEG HDF5 arrays or the channel-level Parquet features, and every answer is calculated from stored numerical evidence rather than inferred subjectively from pixels.

The original dataset source and license are documented in `README_DATA.md`. These generated images and large JSONL files are data artifacts and must not be committed to Git.

## Subject isolation

The original subject-safe split is unchanged:

- Train: S001–S020
- Validation: S021–S025
- Test: S026–S030

Multimodal SFT and RLVR use train images only. The held-out multimodal evaluation uses test images only. Validation-subject images are generated for pipeline development and calibration but are not included in the train or held-out JSONL files. The integrity checker traces every task through its image metadata to its source EEG sample.

## Image selection and counts

Epochs are selected deterministically and approximately balanced across normalized conditions using SHA-256 ordering: 200 train, 50 validation, and 100 test epochs. Each selected epoch produces four core images. Every fourth selected epoch also produces a waveform image. Three subject-level alpha/mu condition comparisons are generated for each of 30 subjects.

The completed dataset contains 1,578 images:

| Visualization | Count |
|---|---:|
| Four-panel scalp topomap | 350 |
| Spectrogram | 350 |
| Power spectral density | 350 |
| Channel-wise band power | 350 |
| Condition comparison | 90 |
| Waveform | 88 |

Split totals are 910 train, 228 validation, and 440 test images.

## Visualization methodology

### Scalp topomaps

Each image contains delta, theta, alpha/mu, and beta absolute-power maps for the same four-second epoch. Electrode coordinates come from MNE’s standard 10–20 montage with case-insensitive matching to the dataset’s normalized 64-channel labels. MNE topographic interpolation, six contours, sensor markers, and the viridis colormap are fixed. The metadata retains all 64 source values for all four bands.

### Spectrograms

The displayed channel is deterministic: C3 for right-fist epochs, C4 for left-fist epochs, and CZ otherwise. SciPy spectrogram parameters are Hann window, 128 samples per segment, 96-sample overlap, 256-point FFT, density scaling, and a displayed 1–40 Hz range. Metadata contains the peak frequency and the channel’s four precomputed band powers.

### Power spectral density

Welch PSD is calculated independently for C3, CZ, and C4 using 320-sample segments and 160-sample overlap, then averaged arithmetically. The 1–40 Hz curve uses a logarithmic vertical axis and fixed shaded band boundaries. Metadata records peak frequency and group-mean band powers.

### Channel-wise band power

Delta, theta, alpha/mu, and beta power are plotted in acquisition channel order. Metadata retains each channel value and complete deterministic descending rankings for every band.

### Condition comparisons

For each subject, channel-wise mean alpha/mu power is compared for left fist versus right fist, both fists versus both feet, and all non-rest movement versus rest. Every contributing sample ID and both 64-channel vectors are retained.

### Waveforms

The filtered C3, CZ, and C4 signals are shown across the fixed four-second epoch. Vertical display offsets prevent overlap and do not alter the underlying HDF5 arrays. RMS and peak-to-peak values are stored as evidence.

All figures use Matplotlib’s non-interactive backend, fixed dimensions, 100 DPI, fixed labels, and fixed plotting parameters. PNG SHA-256 hashes are stored in image metadata.

## Multimodal tasks

`multimodal_sft_train.jsonl` contains 700 train-only examples with an image reference, researcher question, context, grounded answer, numerical evidence, and source samples. `multimodal_rlvr_train.jsonl` contains 900 train-only tasks. `multimodal_eval_heldout.jsonl` contains all 440 test images and covers topomap, spectrogram, PSD, channel ranking, waveform, and condition-comparison reasoning.

Verifiable operations include argmax channel/band selection, exact top-three channel ranking, exact categorical comparison, and numeric PSD peak-frequency reporting with explicit tolerance. The stored ground truth is independently reconstructed from image metadata during validation.

## Validation and leakage controls

`vision/validate.py` checks unique image IDs; file existence and SHA-256; image path/metadata agreement; duplicate hashes across splits; train/test task, image, and question isolation; finite source evidence; official channel labels; source condition and subject agreement; independently recomputed ground truth; and byte-identical deterministic regeneration for one complete epoch-image group from every split.

The final run passes all 13 end-to-end checks. Fourteen verifier and task-construction unit tests also pass.

## Final training-data quality pass

The finalized corpus contains 700 multimodal SFT examples, 900 vision RLVR examples, and 440 strictly held-out evaluation examples. All task JSONL files use repository-relative image references and retain the source operation and operands needed to recompute every ground truth.

| Corpus | Topomap | PSD | Spectrogram | Band power | Condition comparison | Waveform |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 147 | 148 | 147 | 148 | 60 | 50 |
| RLVR | 197 | 198 | 197 | 198 | 60 | 50 |
| Held-out evaluation | 100 | 100 | 100 | 100 | 15 | 25 |

| Corpus | Numeric | Categorical | Comparison | Ranking | Set/membership |
|---|---:|---:|---:|---:|---:|
| SFT | 135 | 238 | 89 | 152 | 86 |
| RLVR | 168 | 314 | 105 | 199 | 114 |
| Held-out evaluation | 80 | 160 | 43 | 99 | 58 |

Verifier types are tied to explicit operations: numeric identity, maximum, or percent change uses absolute and relative tolerances of `1e-6`; categorical tasks use normalized exact matching; rankings require exact deterministic top-k order with channel-name tie breaking; and set tasks require normalized set equality. Comparison tasks have categorical or numeric verifiers according to their ground-truth type. `vision/finalize.py` independently evaluates the stored operation against its operands for every SFT, RLVR, and evaluation item.

The audit found no broken image references, unsupported answers, invalid tolerances, verifier mismatches, within-corpus duplicate questions, train/evaluation question duplication, or split leakage. SFT and RLVR intentionally contain paired views of 700 train images; this is recorded separately and is not evaluation leakage. All 2,040 task examples were regenerated during the quality pass to improve answer specificity, template diversity, balance, and set-task coverage. No image or final example was removed. The cross-band topomap comparison family was discontinued because the four panels use independent color normalization; retained topomap tasks compare electrodes only within a band.

Manual-review gold files contain 50 representative SFT, 50 RLVR, and 50 held-out examples, balanced across available visual modalities and task classes. Final counts, audit results, detailed distributions, and the portable SHA-256 inventory are stored respectively in `vision_dataset_final_summary.json`, `vision_quality_audit.json`, `vision_task_distribution.json`, and `vision_dataset_package_manifest.json`.

## Reproduction

Run from the project root after completing the base EEG pipeline. Required packages are NumPy, SciPy, pandas, PyArrow, h5py, Matplotlib, Pillow, MNE, and pytest.

```bash
MPLCONFIGDIR=/tmp/neuro_agent_matplotlib python vision/generate.py --limits train=200,validation=50,test=100
MPLCONFIGDIR=/tmp/neuro_agent_matplotlib python vision/build_tasks.py
MPLCONFIGDIR=/tmp/neuro_agent_matplotlib python -m pytest -q tests/test_verifiers.py tests/test_vision_task_ground_truth.py
MPLCONFIGDIR=/tmp/neuro_agent_matplotlib python vision/validate.py
python vision/finalize.py
```

Primary outputs are under `data/processed/vision/`. Image provenance is in `metadata/images.jsonl`, dataset counts are in `metadata/summary.json`, and validation results are in `metadata/validation_report.json`.

## Limitations

Topomaps interpolate sparse electrode measurements and must not be interpreted as direct images of brain sources. Absolute power depends on the recorded montage and acquisition conditions. Spectrogram resolution is constrained by four-second epochs. Task annotations represent instructed states rather than verified limb kinematics. The visual tasks emphasize objective signal-analysis operations and deliberately avoid clinical or unsupported biological interpretation.
