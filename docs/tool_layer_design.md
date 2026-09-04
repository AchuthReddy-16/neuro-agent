# Neuroscience Tool Layer Design

**Status:** Planning only (no implementation). GPU must remain idle for this stage.

**Date:** 2026-09-02

**Problem:** Multimodal VLM is weak on exact numerical and set operations (set membership ~0%, waveform numeric weak, peak-frequency weak). A deterministic tool layer is required to supply exact evidence instead of visual estimation.

---

## 1. Existing Reusable Tools (Audit)

Audit scope: `src/`, `scripts/`, `data/`, `results/`, `benchmarks/`.

Legend:
- **EXISTS AND REUSABLE** — callable logic exists; minor wrapping for agent use
- **EXISTS BUT NEEDS REFACTOR** — partial logic; must be extracted/refactored into tools
- **MISSING** — no implementation in `src/`; may have offline precomputed artifacts only

| # | Capability | Status | Exact paths | Notes |
|---|------------|--------|-------------|-------|
| 1 | Band-power | **EXISTS BUT NEEDS REFACTOR** | Precomputed: `data/processed/features.parquet` (`delta_power`, `theta_power`, `alpha_mu_power`, `beta_power` per channel row). Baseline features: `data/processed/baseline/{split}.npz` (`feature_names` include `{band}__{channel}`). SFT context pattern: `scripts/build_sft_corrective_v2_mixed_dataset.py` → `tool_context.operation = "computed_feature_lookup"`. | No runtime `compute_band_power()` in `src/`. Must recompute from epoch HDF5 or read precomputed features with a stable API. |
| 2 | RMS | **EXISTS BUT NEEDS REFACTOR** | Precomputed: `features.parquet` column `rms` (µV). Eval context: `evaluation/verifiers.py` consumers use `context.values` channel→RMS maps. Waveform evidence: `data/processed/vision/metadata/images.jsonl` → `source_numeric_values.rms_uV`. | No `compute_rms()` in `src/`. |
| 3 | PSD / peak-frequency | **MISSING** (runtime) | Documented offline: `README_VISION_DATA.md` (Welch: 320-sample segments, 160 overlap, C3/CZ/C4 average; spectrogram: Hann 128, overlap 96, FFT 256). Stored evidence: vision `images.jsonl` → `source_numeric_values.peak_frequency_hz`. Tasks: `spectrogram_peak_frequency`, `psd_peak_frequency` in multimodal eval. | Pipeline scripts (`vision/generate.py`) referenced in README but **not present in git repo**. |
| 4 | Channel ranking | **EXISTS BUT NEEDS REFACTOR** | Response parsing only: `src/neuro_agent/evaluation/verifiers.py` → `parse_ranking()`, `verify_example()` (`verification_type == "ranking"`). Graded reward: `src/neuro_agent/training/rewards.py` → `ranking_graded_reward()`. Precomputed rankings: vision metadata `channel_rankings_descending`. | Parses model output; does not rank channels from data. |
| 5 | Threshold / set membership | **EXISTS BUT NEEDS REFACTOR** | Parsing: `verifiers.py` → `parse_set_membership()`, `verify_example()` (`verification_type == "set"`). Reward: `rewards.py` → `set_graded_reward()`. SFT pattern: `sft_train.jsonl` threshold questions with `tool_context.inputs.threshold` + `beta_power_uV2`. Multimodal: `source_values.operation = "at_or_above_threshold"`. | No `select_channels_above_threshold()` implementation. Base VLM set pass rate 0% on held-out multimodal eval. |
| 6 | Condition comparison | **EXISTS BUT NEEDS REFACTOR** | Precomputed: `data/processed/condition_summaries.parquet` (per split/task/movement/condition/channel aggregates). Vision: `images.jsonl` comparison type → `condition_a`, `condition_b`, `mean_a`, `mean_b`, `channel_values_a/b`, `largest_absolute_difference_channel`. Eval tasks: `condition_higher_mean`, `condition_largest_channel_difference`. | No runtime compare tool in `src/`. |
| 7 | Percent difference | **MISSING** (runtime) | Task family: `condition_percent_difference` in `data/metadata/vision_corrective/training_data_audit.json`. Multimodal base eval pass rate 0% (`results/multimodal_base_eval/per_task_metrics.json`). | Formula implied by task ground truth; no function in `src/`. |
| 8 | Effect size | **MISSING** (runtime) | Precomputed: `data/processed/effect_sizes.parquet` columns: `split`, `movement`, `channel`, `metric`, `condition_a`, `condition_b`, `cohens_d`, `n_a`, `n_b`. | Offline batch only; no lookup/compute API. |
| 9 | Correlation | **MISSING** (runtime) | Precomputed: `data/processed/correlations.parquet` columns: `split`, `channel`, `variable_x`, `variable_y`, `pearson_r`, `n`. | Offline batch only. |
| 10 | Classifier inference | **EXISTS AND REUSABLE** | `src/neuro_agent/evaluation/eeg_classifier.py` → `run_baseline_classifier()`, `_build_models()`, `_evaluate()`. Data loader: `src/neuro_agent/data/eeg_baseline.py` → `load_split()`, `EegSplit`. Entry script: `scripts/run_eeg_baseline.py`. | Trains/evaluates batch sklearn models on 576-d tabular features. Needs thin `predict_movement(sample_id)` wrapper; not epoch-level inference. |
| 11 | Outlier detection | **MISSING** | No references in `src/`, `scripts/`, or processed artifacts. | Required for V1 only if agent must flag anomalous epochs; otherwise defer. |
| 12 | Topomap generation | **MISSING** (in repo) | Artifacts: `data/processed/vision/images/` (PNG), metadata `visualization_type = topomap_multi_band`. Documented: `README_VISION_DATA.md`. | Generation code not in git; images + `source_numeric_values` are reusable evidence sources. |
| 13 | Spectrogram generation | **MISSING** (in repo) | Artifacts: vision images `visualization_type = spectrogram`, metadata `peak_frequency_hz`, `band_powers`, `channel`. | Same as topomap — offline artifacts only. |

### Supporting infrastructure (not tools, but reusable)

| Component | Path | Role |
|-----------|------|------|
| Tool ABC stub | `src/neuro_agent/tools/__init__.py` | `Tool`, `LiteratureSearchTool`, `DataAnalysisTool` (NotImplemented) |
| Agent stub | `src/neuro_agent/agents/__init__.py` | `NeuroResearchAgent` (NotImplemented) |
| Verifiers | `src/neuro_agent/evaluation/verifiers.py` | Deterministic answer checking (numeric, categorical, set, ranking) |
| RLVR rewards | `src/neuro_agent/training/rewards.py` | Graded rewards mirroring verifiers |
| Multimodal eval normalize | `src/neuro_agent/multimodal/dataset.py` | `normalize_eval_example()`, `TASK_CLASS_TO_VERIFIER` |
| Web API types (future) | `web/src/lib/types.ts` | `ToolInvocation`, `EvidenceItem`, `AnalyzeRequest.tools` |

### Benchmarks directory

`benchmarks/` contains inference/KV-cache JSON summaries only — no neuroscience tool implementations.

---

## 2. Data Interface Schemas (Measured)

All shapes verified from on-disk artifacts under `data/processed/` (CPU schema inspection only).

### 2.1 Epoch registry — `samples.jsonl`

One JSON object per line, 11,700 epochs.

| Field | Type | Example / shape |
|-------|------|-----------------|
| `sample_id` | str | `S001_R01_E000` |
| `subject_id` | str | `S001` |
| `run_id` | str | `R01` |
| `task_type` | str | `baseline` \| `execution` \| `imagery` |
| `movement` | str | `rest` \| `left_fist` \| `right_fist` \| `both_fists` \| `both_feet` |
| `condition` | str | `baseline_rest`, `execution_left_fist`, … |
| `protocol` | str | `eyes_open` |
| `event_code` | str | `T0`, `T1`, … |
| `start_time`, `end_time` | float | seconds within run (epoch window) |
| `sampling_rate` | float | **160.0 Hz** (all recordings) |
| `n_channels` | int | **64** |
| `n_samples` | int | **640** (= 4.0 s × 160 Hz) |
| `array_path` | str | `data/processed/arrays/S001_R01.h5` |
| `array_index` | int | epoch index within HDF5 `eeg` dataset |
| `feature_path` | str | `data/processed/features.parquet` |
| `split` | str | `train` \| `validation` \| `test` |
| `metadata` | object | `epoch_seconds: 4.0`, `filter_hz: [1, 40]`, `source_file` |

**Channel names (normalized, 64):** see `data/metadata/preprocessing_report.json` → `channel_names` (e.g. `FC5`, `C3`, `FP1`, …).

**Filter:** 4th-order zero-phase Butterworth, passband 1–40 Hz (`preprocessing_report.json`).

### 2.2 Raw epoch arrays — `arrays/{subject}_{run}.h5`

| Dataset | Shape | Dtype | Notes |
|---------|-------|-------|-------|
| `eeg` | `(n_epochs, 64, 640)` | float32 | Units: µV (`attrs.units`) |
| `channel_names` | `(64,)` | bytes/str | Matches normalized 10–20 labels |
| attrs `sampling_rate` | scalar | float | 160.0 |

Example: `S001_R01.h5` → `eeg.shape = (15, 64, 640)`.

### 2.3 Per-channel features — `features.parquet`

Long format: **748,800 rows** = 11,700 samples × 64 channels.

| Column | Type | Units / notes |
|--------|------|----------------|
| `sample_id`, `subject_id`, `run_id`, `split`, `task_type`, `movement`, `condition` | str | Join keys |
| `channel` | str | One of 64 electrode names |
| `mean` | float | µV |
| `variance` | float | µV² |
| `std` | float | µV |
| `rms` | float | µV |
| `peak_to_peak` | float | µV |
| `delta_power` | float | µV² (band-limited power) |
| `theta_power` | float | µV² |
| `alpha_mu_power` | float | µV² |
| `beta_power` | float | µV² |

### 2.4 Baseline classifier matrices — `baseline/{split}.npz`

| Array | Shape (train) | Description |
|-------|---------------|-------------|
| `X` | `(7800, 576)` | float64 standardized features |
| `y` | `(7800,)` | movement labels: `rest`, `left_fist`, … |
| `sample_ids` | `(7800,)` | e.g. `S001_R01_E000` |
| `feature_names` | `(576,)` | `{mean,variance,std,rms,peak_to_peak,delta_power,theta_power,alpha_mu_power,beta_power}__{CHANNEL}` |

Metadata sidecar: `baseline/{split}_metadata.parquet` → `sample_id`, `subject_id`, `run_id`, `task_type`, `movement`, `condition`.

### 2.5 Aggregated analytics tables

**`effect_sizes.parquet`** — (960, 9): `split`, `movement`, `channel`, `metric`, `condition_a`, `condition_b`, `cohens_d`, `n_a`, `n_b`.

**`correlations.parquet`** — (192, 6): `split`, `channel`, `variable_x`, `variable_y`, `pearson_r`, `n`.

**`condition_summaries.parquet`** — (2112, 32): grouped means/stds/counts per `split`, `task_type`, `movement`, `condition`, `channel` for all scalar metrics.

### 2.6 Text eval / SFT / RLVR task schemas

**`eval_heldout.jsonl`** (1,000 test examples):

```json
{
  "id": "eval_…",
  "category": "channel_ranking | band_power_analysis | numerical_reasoning | …",
  "question": "…",
  "context": { "values": { "AF3": 116.34, … } } | { "channel": "AF3", "variance": 3911.75 },
  "ground_truth": …,
  "verification_type": "numeric | categorical | ranking | set",
  "tolerance": { "absolute": 1e-6, "relative": 1e-6 },
  "source_samples": ["S027_R13_E005"]
}
```

**`sft_train.jsonl`** (2,400 examples):

```json
{
  "id": "…",
  "question": "…",
  "answer": "…",
  "tool_context": {
    "operation": "computed_feature_lookup",
    "inputs": { "beta_power_uV2": { "C3": 6208.96, … }, "threshold": 123.4 }
  },
  "grounded_facts": { "ground_truth": …, "verification_type": "ranking" },
  "source_samples": ["S003_R07_E012"]
}
```

**`rlvr_train.jsonl`** (3,269 examples): adds top-level `task_type`, `verification_type`, `context` (often full channel dicts), `tolerance`.

### 2.7 Vision multimodal schemas

**`vision/metadata/images.jsonl`** (1,578 images):

| `visualization_type` | Key `source_numeric_values` |
|----------------------|----------------------------|
| `channel_band_power` | `delta/theta/alpha_mu/beta_power` (64-ch dicts), `channel_rankings_descending` |
| `waveform` | `rms_uV`, `peak_to_peak_uV` (C3, CZ, C4) |
| `spectrogram` | `channel`, `peak_frequency_hz`, `band_powers`, `dominant_band` |
| `power_spectral_density` | `peak_frequency_hz`, `group_band_powers`, `dominant_band` |
| `topomap_multi_band` | four band dicts (64 ch each) |
| `condition_comparison` | `condition_a/b`, `mean_a/b`, `channel_values_a/b`, `largest_absolute_difference_channel` |

**`vision/multimodal_eval_heldout.jsonl`** (440 examples):

```json
{
  "id": "veval_…",
  "image_id": "img_…",
  "image_path": "data/processed/vision/images/…",
  "task_family": "band_power_high_beta_set",
  "task_class": "set_membership",
  "verification_type": "set",
  "question": "…",
  "context": { "subject_id", "run_id", "condition", "visualization_type" },
  "source_values": {
    "operation": "at_or_above_threshold | identity | argmax | …",
    "values": { "FC5": 25.81, … },
    "k": 3,
    "units": "Hz"
  },
  "ground_truth": ["AF3", "AF4", …] | 2.5 | "delta_power",
  "tolerance": { "absolute": 1e-6, "relative": 1e-6 },
  "source_samples": ["S027_R13_E008"]
}
```

### 2.8 Subject splits

| Split | Subjects | Samples |
|-------|----------|---------|
| train | S001–S020 | 7,800 |
| validation | S021–S025 | 1,950 |
| test | S026–S030 | 1,950 |

Enforced by: `eeg_baseline.verify_subject_splits()`, `llm_eval.verify_heldout_integrity()`, multimodal dataset loaders.

---

## 3. Proposed Tool Contracts (API Design Only)

All tools live under future package `src/neuro_agent/tools/eeg/`. Each returns a structured `ToolResult`:

```python
@dataclass
class ToolResult:
    tool: str
    success: bool
    value: Any              # primary answer payload
    evidence: dict[str, Any]  # numeric audit trail
    units: str | None
    sample_ids: list[str]
    error: str | None = None
```

Shared validation:
- `sample_id` must match `^(S\d{3})_(R\d{2})_(E\d{3})$`
- `channel` ∈ official 64-channel list (`preprocessing_report.json`)
- `band` ∈ `{delta, theta, alpha_mu, beta}` → maps to `{delta_power, theta_power, alpha_mu_power, beta_power}`
- Reject cross-split leakage when `forbidden_subjects` provided (agent eval mode)

Tolerance defaults (match verifiers): `absolute=1e-6`, `relative=1e-6`; peak-frequency may use `absolute=0.5` Hz minimum band per RLVR shaping.

---

### `compute_band_power`

| | |
|--|--|
| **Purpose** | Absolute band-limited power for one or all channels in an epoch |
| **Input** | `sample_id: str`, `band: Band`, `channels: list[str] \| "all"`, `source: "features" \| "recompute" = "features"` |
| **Output** | `{channel: float}` dict or single float; `evidence.method`, `evidence.filter_hz` |
| **Units** | µV² |
| **Validation** | Sample exists; band known; channels ⊆ 64 |
| **Errors** | `SampleNotFound`, `ChannelNotFound`, `FeatureMissing` |

---

### `compute_rms`

| | |
|--|--|
| **Purpose** | RMS amplitude for channel(s) in an epoch |
| **Input** | `sample_id: str`, `channels: list[str] \| "all"` |
| **Output** | `{channel: float}` or float |
| **Units** | µV |
| **Validation** | Same as above |
| **Errors** | `SampleNotFound`, `ChannelNotFound` |

---

### `compute_psd` / `find_psd_peak`

**`compute_psd`**

| | |
|--|--|
| **Purpose** | Welch PSD for specified channels |
| **Input** | `sample_id`, `channels: list[str]` (default `["C3","CZ","C4"]`), `fmin=1`, `fmax=40`, `nperseg=320`, `noverlap=160` |
| **Output** | `{freqs_hz: list[float], psd_uv2_hz: {ch: list[float]}, group_mean: list[float]}` |
| **Units** | µV²/Hz |
| **Validation** | Params match vision pipeline when `match_vision_pipeline=True` |
| **Errors** | `SampleNotFound`, `InvalidFrequencyRange` |

**`find_psd_peak`**

| | |
|--|--|
| **Purpose** | Argmax frequency of group-mean PSD in band |
| **Input** | `sample_id`, same PSD params, `fmin`, `fmax` |
| **Output** | `value: float` (Hz), `evidence.psd_peaks_per_channel` |
| **Units** | Hz |
| **Validation** | `fmax > fmin`; tolerance for near-tie peaks documented |
| **Errors** | `FlatSpectrum`, `SampleNotFound` |

---

### `compute_spectrogram_peak`

| | |
|--|--|
| **Purpose** | Peak displayed frequency from spectrogram pipeline (single channel rule: C3/C4/CZ) |
| **Input** | `sample_id`, `fmin=1`, `fmax=40`, `nperseg=128`, `noverlap=96`, `nfft=256` |
| **Output** | `{channel: str, peak_frequency_hz: float, band_powers: dict}` |
| **Units** | Hz |
| **Validation** | Channel selection rule from movement condition |
| **Errors** | `SampleNotFound`, `SpectrogramComputationError` |

---

### `rank_channels`

| | |
|--|--|
| **Purpose** | Deterministic descending rank by metric |
| **Input** | `sample_id`, `metric: str` (e.g. `rms`, `beta_power`), `top_k: int \| None`, `channels: list[str] \| "all"` |
| **Output** | `{ranking: list[str], values: {ch: float}}` |
| **Units** | Inherited from metric |
| **Validation** | Tie-break: ascending channel name (matches vision `channel_rankings_descending`) |
| **Errors** | `UnknownMetric`, `SampleNotFound` |

---

### `select_channels_above_threshold`

| | |
|--|--|
| **Purpose** | Set membership: channels where `value {op} threshold` |
| **Input** | `values: dict[str,float]` **or** `sample_id + metric`, `threshold: float`, `op: "gt" \| "ge"` (default `gt`), `threshold_mode: "absolute" \| "median" \| "upper_quartile"` |
| **Output** | `{channels: list[str], threshold_used: float, n_selected: int}` |
| **Units** | Same as metric |
| **Validation** | Threshold computed deterministically from supplied value dict when mode ≠ absolute |
| **Errors** | `EmptyValueDict`, `InvalidThreshold` |

---

### `compare_conditions`

| | |
|--|--|
| **Purpose** | Compare two conditions for a subject (alpha/mu mean or arbitrary metric) |
| **Input** | `subject_id`, `condition_a`, `condition_b`, `metric: str = "alpha_mu_power"`, `aggregation: "mean_across_epochs"` |
| **Output** | `{mean_a, mean_b, channel_values_a: dict, channel_values_b: dict, largest_abs_diff_channel}` |
| **Units** | µV² (for power metrics) |
| **Validation** | Both conditions exist for subject in `condition_summaries` or computable from samples |
| **Errors** | `ConditionNotFound`, `InsufficientSamples` |

---

### `compute_percent_difference`

| | |
|--|--|
| **Purpose** | Signed percent change `(b - a) / a * 100` |
| **Input** | `a: float`, `b: float` **or** `subject_id`, `condition_a`, `condition_b`, `metric` |
| **Output** | `{percent: float, a, b}` |
| **Units** | percent (dimensionless) |
| **Validation** | Reject `a == 0` unless explicit `zero_baseline_policy` |
| **Errors** | `ZeroBaseline`, `ConditionNotFound` |

---

### `compute_effect_size`

| | |
|--|--|
| **Purpose** | Cohen's d between two conditions for a channel/metric |
| **Input** | `split`, `movement`, `channel`, `metric`, `condition_a`, `condition_b`, `source: "table" \| "recompute"` |
| **Output** | `{cohens_d: float, n_a, n_b}` |
| **Units** | dimensionless |
| **Validation** | Lookup first in `effect_sizes.parquet` |
| **Errors** | `PairNotFound`, `InsufficientSamples` |

---

### `compute_correlation`

| | |
|--|--|
| **Purpose** | Pearson r between two metrics (per channel, split-wide) |
| **Input** | `split`, `channel`, `variable_x`, `variable_y` |
| **Output** | `{pearson_r: float, n: int}` |
| **Units** | dimensionless |
| **Validation** | Lookup in `correlations.parquet` or recompute from `features.parquet` |
| **Errors** | `PairNotFound`, `ConstantSeries` |

---

### `predict_movement`

| | |
|--|--|
| **Purpose** | Classical movement-state classification from tabular features |
| **Input** | `sample_id`, `model: "logistic_regression" \| "random_forest" \| "xgboost"`, `return_proba: bool = False` |
| **Output** | `{label: str, proba: dict \| None, model_name, feature_source}` |
| **Units** | categorical |
| **Validation** | Model trained on train split only; scaler from train |
| **Errors** | `ModelNotLoaded`, `SampleNotFound` |

---

### `detect_outliers` (optional V2)

| | |
|--|--|
| **Purpose** | Flag epochs/channels with z-score or IQR above threshold |
| **Input** | `sample_id` or `subject_id`, `metric`, `method: "zscore" \| "iqr"`, `threshold: float` |
| **Output** | `{outliers: list[{sample_id, channel, value, score}]}` |
| **Errors** | `InsufficientReferenceData` |

---

### `generate_topomap` / `generate_spectrogram` (optional V2)

| | |
|--|--|
| **Purpose** | Regenerate PNG + metadata for agent evidence (not V1) |
| **Input** | `sample_id`, visualization params matching `README_VISION_DATA.md` |
| **Output** | `{image_path, source_numeric_values, sha256}` |
| **Errors** | `MNEUnavailable`, `RenderError` |

---

## 4. Question → Tool Routing Map (Design Only)

| Question pattern / task family | Primary tool(s) | Secondary / evidence |
|-------------------------------|-----------------|----------------------|
| Highest/lowest band power channel | `rank_channels(metric=beta_power, top_k=1)` | `compute_band_power` |
| Top-k band power ranking | `rank_channels(top_k=k)` | `compute_band_power` |
| RMS for channel X | `compute_rms(channels=[X])` | features lookup |
| Max RMS among {C3,CZ,C4} | `rank_channels(metric=rms, channels=[C3,CZ,C4], top_k=1)` | `compute_rms` |
| RMS ordering | `rank_channels(metric=rms, top_k=3)` | waveform values |
| PSD peak frequency (1–40 Hz) | `find_psd_peak` | `compute_psd` |
| Spectrogram peak frequency | `compute_spectrogram_peak` | — |
| Dominant band (PSD/spectrogram) | `rank_channels` on band power dict **or** `compute_band_power` + argmax | precomputed vision metadata |
| Band ordering (4 bands) | `rank_channels` on per-band aggregates | `compute_band_power(all)` |
| Channels above median/quartile threshold | `select_channels_above_threshold` | `compute_band_power` |
| Set membership (high beta/delta) | `select_channels_above_threshold` | — |
| Which condition higher mean | `compare_conditions` | — |
| Largest channel-wise condition difference | `compare_conditions` → `largest_abs_diff_channel` | — |
| Percent difference between conditions | `compute_percent_difference` | `compare_conditions` |
| Effect size between conditions | `compute_effect_size` | `compare_conditions` |
| Correlation between metrics | `compute_correlation` | — |
| Movement label / MI classification | `predict_movement` | sample metadata lookup |
| Normalized condition label | metadata lookup from `samples.jsonl` (not visual) | — |
| Variance / mean / std numeric | features row lookup | `compute_rms` if RMS |
| Outlier / artifact screening | `detect_outliers` (V2) | — |
| “Show topomap/spectrogram” | return existing `image_path` from vision index (V2: `generate_*`) | — |

**Routing policy:** If question includes image reference (`image_id` / `img_*`), resolve `epoch_sample_id` from `vision/metadata/images.jsonl`, then prefer **recompute from `sample_id`** over reading pixels. Use precomputed `source_numeric_values` only as cross-check.

---

## 5. Minimum Tool Set

### Mandatory V1 (smallest set addressing VLM failures)

| Tool | Rationale |
|------|-----------|
| `compute_band_power` | Core evidence for ranking, argmax, dominant-band |
| `compute_rms` | Waveform numeric failures |
| `find_psd_peak` | PSD peak-frequency 0–3% pass rate |
| `compute_spectrogram_peak` | Spectrogram peak-frequency 0% pass rate |
| `rank_channels` | Channel ranking / top-k |
| `select_channels_above_threshold` | Set membership 0% pass rate |
| `compare_conditions` | Condition comparison tasks |
| `compute_percent_difference` | Numeric comparison failures |
| `lookup_sample_metadata` | Non-visual factual grounding (task_type, movement, condition) |
| `resolve_vision_evidence` | Map `image_id` → `sample_id` + `source_numeric_values` |

### Optional later

| Tool | Defer reason |
|------|--------------|
| `compute_effect_size` | Precomputed table exists; lower eval frequency |
| `compute_correlation` | Precomputed table exists |
| `predict_movement` | Separate sklearn path; tabular not image |
| `detect_outliers` | No current eval task family |
| `generate_topomap` / `generate_spectrogram` | Heavy deps; images already on disk |
| `compute_psd` (full curve) | Only if peak tool insufficient for debugging |
| Literature / generic `DataAnalysisTool` | Out of neuroscience numeric scope |

---

## 6. Test Plan (Design Only)

Per-tool unit tests under `tests/tools/` (pytest, CPU only). No full model benchmarks.

| Tool | Unit tests | Tolerance | Subject leakage | Example I/O |
|------|------------|-----------|-----------------|-------------|
| `compute_band_power` | Match `features.parquet` for 10 random `sample_id`×channel; recompute vs table | rel 1e-5 on power | Train tool fit on S001–S020 only | `S001_R01_E000`, beta, C3 → finite µV² |
| `compute_rms` | Match features `rms` column | abs 1e-4 µV | — | `S026_R03_E010`, C3 → scalar |
| `find_psd_peak` | Match vision `images.jsonl` `peak_frequency_hz` for 5 PSD images | abs 0.01 Hz | — | peak ≈ 1.5 Hz |
| `compute_spectrogram_peak` | Match vision spectrogram metadata | abs 0.01 Hz | — | peak ≈ 2.5 Hz |
| `rank_channels` | Exact order vs `channel_rankings_descending` in metadata | exact channel names | — | top-3 beta matches JSON |
| `select_channels_above_threshold` | Match multimodal `ground_truth` for 10 set tasks | exact set equality | — | upper-quartile beta set |
| `compare_conditions` | Match comparison image `mean_a/b` | rel 1e-5 | Never aggregate across test subjects in training fixtures | left_fist vs right_fist |
| `compute_percent_difference` | Match `condition_percent_difference` ground truth | abs 1e-4 % | — | signed % from means |
| `lookup_sample_metadata` | Fields match `samples.jsonl` | exact strings | Held-out S026–S030 readable but flagged | `imagery_rest` |
| `resolve_vision_evidence` | `image_id` → correct `epoch_sample_id` | exact | Eval images only S026–S030 | `img_psd_*` → sample |
| `predict_movement` | Confusion matrix subset vs `eeg_classifier` | accuracy ≥ baseline on 50 test rows | Model trained train only | label ∈ 5 classes |
| `compute_effect_size` | Match `effect_sizes.parquet` row | abs 1e-4 on d | split-aware | cohens_d numeric |
| `compute_correlation` | Match `correlations.parquet` | abs 1e-4 on r | split-aware | pearson_r |

**Integration tests (lightweight):**
- Agent router maps 20 held-out questions to expected tool chain (mock LLM).
- Tool output fed through `verify_example()` → pass rate 100% on golden set.
- End-to-end: `resolve_vision_evidence` + `select_channels_above_threshold` reproduces 5 failing `band_power_high_beta_set` IDs from `results/multimodal_base_eval/failures.jsonl`.

**Regression guard:** Pin pipeline constants to `README_VISION_DATA.md` and `preprocessing_report.json`.

---

## 7. Implementation Order

1. **Data access layer** — `SampleStore`, `FeatureStore`, `VisionIndex` (read-only wrappers over jsonl/parquet/h5)
2. **V1 numeric tools** — `compute_band_power`, `compute_rms`, `rank_channels`
3. **V1 spectral tools** — `find_psd_peak`, `compute_spectrogram_peak` (port constants from README; restore or vendor vision math)
4. **V1 set/threshold** — `select_channels_above_threshold`
5. **V1 comparison** — `compare_conditions`, `compute_percent_difference`
6. **Metadata / vision bridge** — `lookup_sample_metadata`, `resolve_vision_evidence`
7. **Tool registry + `Tool` subclasses** — wire into `src/neuro_agent/tools/`
8. **Agent router** — separate from this doc
9. **V2** — `predict_movement`, effect size, correlation, outlier, renderers

---

## 8. Effort Estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| Design | This audit + design | **Done** |
| Tools | Data access + V1 tools (8 tools) + unit tests | 4–6 dev-days |
| Router | Agent router + tool-calling loop + evidence assembly | 3–4 dev-days |
| Vision | Vision pipeline code restore (if recompute required) | 2–3 dev-days |
| Eval | Integration with eval harness + ablation vs VLM-only | 2 dev-days |

**Total to production V1 tool layer:** ~11–15 dev-days.

---

## 9. Blockers

| Blocker | Severity | Mitigation |
|---------|----------|------------|
| **Vision generation code not in git** (`vision/generate.py`, `build_tasks.py`, etc.) | High | Vendor from backup or reimplement from `README_VISION_DATA.md` constants; V1 can use `features.parquet` + stored metadata for most tools |
| **Preprocessing pipeline scripts not in repo** (`data/scripts/` empty) | Medium | V1 reads precomputed artifacts; document recompute path for fresh data |
| **No `src/neuro_agent/tools/eeg/` package** | Low | Tools live under `src/neuro_agent/tools/` |
| **Classifier is batch sklearn, not persisted predict API** | Medium | Save fitted scaler+model to `checkpoints/baseline_classifier/` in the tools milestone |
| **Label space mismatch** (movement vs task_type vs compound condition) | Medium | `lookup_sample_metadata` must expose canonical fields; router uses metadata not vision |
| **h5py/MNE/scipy version pinning** | Low | Lock in `pyproject.toml`; test tolerances |
| **Set tasks need exact threshold policy** | High | Encode `median` / `upper_quartile` exactly as vision task builder; golden tests from `source_values` |

---

## 10. Summary Tables

### Reusable today

- Verifier parsing/grading: `verifiers.py`, `rewards.py`
- Classifier training/eval: `eeg_classifier.py`, `eeg_baseline.py`
- Precomputed features & analytics parquet
- Vision image metadata + PNG artifacts
- Multimodal eval normalization: `multimodal/dataset.py`

### Missing (must build)

- All deterministic compute/lookup tools in `src/neuro_agent/tools/`
- Agent tool router and `NeuroResearchAgent.run()`
- Persisted sklearn inference endpoint
- Outlier detection
- In-repo vision/preprocessing regeneration code

### Multimodal base eval weak families (motivation)

From `results/multimodal_base_eval/per_task_metrics.json`:

| Task family | Pass rate |
|-------------|-----------|
| `band_power_high_beta_set` | 0% |
| `topomap_high_delta_set` | 0% |
| `spectrogram_peak_frequency` | 0% |
| `psd_peak_frequency` | ~3% |
| `condition_percent_difference` | 0% |
| `waveform_max_rms_numeric` | 62.5% |

---

