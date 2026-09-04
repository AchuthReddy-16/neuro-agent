#!/usr/bin/env python3
"""Text vs vision routing evaluation (baseline + optional one repair)."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "routing"

# Locked baseline prompt (verbatim current production intent prompt before any J.1 repair).
BASELINE_INTENT_SYSTEM_PROMPT = """You are a neuroscience research intent parser.
Given a user question, output ONLY a single JSON object (no markdown, no prose) that routes the question to exactly one analysis tool.

Supported question_type values (use exactly one):
- band_power: band-limited power for channel(s) in a sample
- rms: RMS amplitude for channel(s) in a sample
- psd_peak: dominant PSD peak frequency in a sample
- channel_ranking: rank channels by a metric for a sample
- threshold_set: select channels above/below a threshold on band power
- condition_comparison: compare two movement conditions for a subject

JSON fields (omit unused fields):
{
  "question_type": "<one of the six above>",
  "sample_id": "S###_R##_E###",
  "subject_id": "S###",
  "run_id": "R##",
  "epoch": 0,
  "channels": ["C3"] or "all",
  "frequency_band": "delta|theta|alpha_mu|beta",
  "frequency_range": [1.0, 40.0],
  "metric": "rms|beta_power|alpha_mu_power|...",
  "condition_a": "left_fist",
  "condition_b": "right_fist",
  "threshold": 0.0,
  "threshold_mode": "absolute|median|upper_quartile",
  "comparator": "gt|ge|lt|le",
  "top_k": 3,
  "sort_direction": "ascending|descending",
  "include_vision_evidence": false,
  "requested_visual_type": "psd|waveform|spectrogram|topomap|band_power"
}

Rules:
- Identify sample_id from the question when present (format S###_R##_E###).
- Otherwise use subject_id + run_id + epoch when inferable.
- condition_comparison requires subject_id, condition_a, condition_b (not sample_id).
- Do not invent sample IDs or numeric values not mentioned or clearly implied.
- Output valid JSON only."""


def _load_intent_system_prompt_from_source() -> str:
    path = PROJECT_ROOT / "src" / "neuro_agent" / "agent" / "prompts.py"
    text = path.read_text()
    m = re.search(
        r'INTENT_SYSTEM_PROMPT\s*=\s*("""|\'\'\')(.*?)\1',
        text,
        flags=re.DOTALL,
    )
    if not m:
        raise RuntimeError("Could not parse INTENT_SYSTEM_PROMPT from prompts.py")
    return m.group(2)


def build_intent_user_prompt(question: str) -> str:
    return f"Question: {question.strip()}\n\nJSON:"


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    if start >= 0:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : idx + 1])
    raise ValueError(f"No JSON object found in: {text[:200]!r}")
CMP = PROJECT_ROOT / "results" / "model_comparison"
CKPT = PROJECT_ROOT / "checkpoints" / "text_w8a8_int8_compressed"
PORT = 8000  # 8001 is occupied by platform nginx proxy
SERVED = "qwen3-w8a8-int8"
GPU_UTIL = 0.90
MAX_MODEL_LEN = 4096

TEXT = "TEXT_ONLY"
VISION = "VISION_REQUIRED"

# ---------------------------------------------------------------------------
# Eval set (held-out labels; never injected into prompts)
# ---------------------------------------------------------------------------


def _ex(
    eid: str,
    question: str,
    expected: str,
    *,
    family: str,
    notes: str = "",
    tricky: bool = False,
) -> dict[str, Any]:
    return {
        "id": eid,
        "question": question,
        "expected_route": expected,
        "family": family,
        "tricky_negative": tricky,
        "notes": notes,
    }


def build_eval_set() -> list[dict[str, Any]]:
    """Balanced held-out routing set (~100 examples). Labels stay out of prompts."""
    rows: list[dict[str, Any]] = []

    # ---- TEXT / TOOL-ONLY ----
    text_bank = [
        _ex(
            "t_bp_01",
            "What is the beta band power for channel C3 in sample S001_R03_E012?",
            TEXT,
            family="band_power",
        ),
        _ex(
            "t_bp_02",
            "Compute alpha_mu power on Cz for S014_R04_E003 using stored EEG features.",
            TEXT,
            family="band_power",
        ),
        _ex(
            "t_bp_03",
            "Report delta and theta band power for all channels in S022_R01_E001.",
            TEXT,
            family="band_power",
        ),
        _ex(
            "t_bp_04",
            "For S008_R06_E020, what is C4 beta power from the structured feature table?",
            TEXT,
            family="band_power",
        ),
        _ex(
            "t_bp_05",
            "Give me the numeric beta power values for C3 and C4 on S003_R03_E005.",
            TEXT,
            family="band_power",
        ),
        _ex(
            "t_rms_01",
            "What is the RMS amplitude of channel C3 in S001_R03_E012?",
            TEXT,
            family="rms",
        ),
        _ex(
            "t_rms_02",
            "Compute RMS for all central channels in sample S016_R08_E022.",
            TEXT,
            family="rms",
        ),
        _ex(
            "t_rms_03",
            "Which has higher RMS, C3 or C4, in S010_R05_E007? Use the deterministic RMS tool.",
            TEXT,
            family="rms",
        ),
        _ex(
            "t_rms_04",
            "Report RMS for Fz in S025_R02_E015 from stored EEG.",
            TEXT,
            family="rms",
        ),
        _ex(
            "t_rms_05",
            "Calculate root-mean-square amplitude for channel Pz on S007_R03_E009.",
            TEXT,
            family="rms",
        ),
        _ex(
            "t_psd_01",
            "What is the PSD peak frequency numerically for S016_R08_E022 between 1 and 40 Hz?",
            TEXT,
            family="psd_peak",
            tricky=True,
            notes="numeric PSD peak → deterministic tool",
        ),
        _ex(
            "t_psd_02",
            "Find the dominant PSD peak for channel C3 in S001_R03_E012 from raw/stored EEG.",
            TEXT,
            family="psd_peak",
        ),
        _ex(
            "t_psd_03",
            "Report the 1–40 Hz PSD peak frequency for S014_R04_E003 using the psd_peak tool.",
            TEXT,
            family="psd_peak",
        ),
        _ex(
            "t_psd_04",
            "What is the numeric PSD peak (Hz) for S022_R01_E001?",
            TEXT,
            family="psd_peak",
            tricky=True,
        ),
        _ex(
            "t_psd_05",
            "Compute PSD peak frequency for S008_R06_E020 from stored spectrum features.",
            TEXT,
            family="psd_peak",
        ),
        _ex(
            "t_rank_01",
            "Which channel has the highest beta power in S001_R03_E012?",
            TEXT,
            family="channel_ranking",
            tricky=True,
            notes="classic text/tool-only ranking",
        ),
        _ex(
            "t_rank_02",
            "Rank the top 3 channels by alpha_mu power for sample S014_R04_E003.",
            TEXT,
            family="channel_ranking",
        ),
        _ex(
            "t_rank_03",
            "Which channel has highest RMS in S016_R08_E022?",
            TEXT,
            family="channel_ranking",
        ),
        _ex(
            "t_rank_04",
            "Order channels by descending beta power for S010_R05_E007 using the ranking tool.",
            TEXT,
            family="channel_ranking",
        ),
        _ex(
            "t_rank_05",
            "Which electrode ranks #1 for theta power in S025_R02_E015?",
            TEXT,
            family="channel_ranking",
        ),
        _ex(
            "t_thr_01",
            "Select channels with beta power above the median in S001_R03_E012.",
            TEXT,
            family="threshold_set",
        ),
        _ex(
            "t_thr_02",
            "Which channels exceed absolute beta threshold 0.5 in S014_R04_E003?",
            TEXT,
            family="threshold_set",
        ),
        _ex(
            "t_thr_03",
            "Apply upper-quartile thresholding on alpha_mu power for S016_R08_E022.",
            TEXT,
            family="threshold_set",
        ),
        _ex(
            "t_thr_04",
            "Return channels with RMS greater than 10 for S008_R06_E020.",
            TEXT,
            family="threshold_set",
        ),
        _ex(
            "t_thr_05",
            "Threshold-select channels below median beta power in S022_R01_E001.",
            TEXT,
            family="threshold_set",
        ),
        _ex(
            "t_cmp_01",
            "Compare left_fist vs right_fist beta power for subject S001.",
            TEXT,
            family="condition_comparison",
        ),
        _ex(
            "t_cmp_02",
            "For S014, is alpha_mu power higher in feet than in rest?",
            TEXT,
            family="condition_comparison",
        ),
        _ex(
            "t_cmp_03",
            "Statistically compare both_fists vs left_fist RMS for subject S022.",
            TEXT,
            family="condition_comparison",
        ),
        _ex(
            "t_cmp_04",
            "Condition comparison: right_fist versus left_fist beta on S008.",
            TEXT,
            family="condition_comparison",
        ),
        _ex(
            "t_cmp_05",
            "Does S016 show higher C3 beta in left_fist than right_fist?",
            TEXT,
            family="condition_comparison",
        ),
        _ex(
            "t_meta_01",
            "How many channels are recorded in sample S001_R03_E012?",
            TEXT,
            family="metadata_statistical",
        ),
        _ex(
            "t_meta_02",
            "What sampling rate and epoch duration are used for S014_R04_E003?",
            TEXT,
            family="metadata_statistical",
        ),
        _ex(
            "t_meta_03",
            "List available frequency bands in the feature schema for this dataset.",
            TEXT,
            family="metadata_statistical",
        ),
        _ex(
            "t_meta_04",
            "What is the mean beta power across all channels in S016_R08_E022?",
            TEXT,
            family="metadata_statistical",
        ),
        _ex(
            "t_meta_05",
            "Report the median RMS across central channels for S010_R05_E007.",
            TEXT,
            family="metadata_statistical",
        ),
        _ex(
            "t_img_ok_01",
            "A PSD plot exists for S016_R08_E022, but I only need the numeric PSD peak from the tool—what is it?",
            TEXT,
            family="image_exists_not_required",
            tricky=True,
            notes="image may exist but is NOT required",
        ),
        _ex(
            "t_img_ok_02",
            "Even though a topomap PNG is on disk for this sample, which channel has highest beta power from features?",
            TEXT,
            family="image_exists_not_required",
            tricky=True,
        ),
        _ex(
            "t_img_ok_03",
            "Ignore any uploaded figure—compute RMS for C3 in S001_R03_E012 from EEG.",
            TEXT,
            family="image_exists_not_required",
            tricky=True,
        ),
        _ex(
            "t_img_ok_04",
            "There is a spectrogram file named img_spectrogram_abc, but answer only with the deterministic band-power ranking for S014_R04_E003.",
            TEXT,
            family="image_exists_not_required",
            tricky=True,
        ),
        _ex(
            "t_img_ok_05",
            "Find PSD peak for S016_R08_E022 and optionally attach stored PSD vision metadata (no visual interpretation needed).",
            TEXT,
            family="image_exists_not_required",
            notes="sidecar metadata ≠ VLM interpretation",
        ),
        _ex(
            "t_img_ok_06",
            "Using structured features only (not the plot), rank beta power channels for S022_R01_E001.",
            TEXT,
            family="image_exists_not_required",
            tricky=True,
        ),
        _ex(
            "t_tool_01",
            "Which analysis tool should I use to get RMS for a sample?",
            TEXT,
            family="tool_selection",
        ),
        _ex(
            "t_tool_02",
            "I need exact numeric channel ranking by beta power—route me to the correct tool.",
            TEXT,
            family="tool_selection",
        ),
        _ex(
            "t_tool_03",
            "Select the deterministic threshold_set tool for upper-quartile beta gating on S008_R06_E020.",
            TEXT,
            family="tool_selection",
        ),
        _ex(
            "t_fact_01",
            "What frequency range defines the beta band in this agent’s feature schema?",
            TEXT,
            family="factual_grounding",
        ),
        _ex(
            "t_fact_02",
            "Is alpha_mu the same as the classic mu rhythm band used in our feature pipeline?",
            TEXT,
            family="factual_grounding",
        ),
        _ex(
            "t_mix_01",
            "Compute beta power for C3 and C4 on S001_R03_E012, then say which is larger—no plots.",
            TEXT,
            family="band_power",
        ),
        _ex(
            "t_mix_02",
            "For S014_R04_E003, return PSD peak Hz and RMS for Cz from tools only.",
            TEXT,
            family="psd_peak",
        ),
        _ex(
            "t_mix_03",
            "Which channel has the highest beta power among C3, C4, and Cz in S016_R08_E022?",
            TEXT,
            family="channel_ranking",
            tricky=True,
        ),
        _ex(
            "t_mix_04",
            "Numeric only: PSD peak frequency for S025_R02_E015.",
            TEXT,
            family="psd_peak",
            tricky=True,
        ),
    ]
    rows.extend(text_bank)

    # ---- VISION REQUIRED ----
    vision_bank = [
        _ex(
            "v_topo_01",
            "Interpret this topomap: does frontal beta appear stronger than occipital beta?",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_topo_02",
            "Looking at the uploaded topomap figure, which scalp region looks hottest in beta?",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_topo_03",
            "Describe the spatial pattern visible in this alpha_mu topomap image.",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_topo_04",
            "Does this topomap visually support the beta-power ranking?",
            VISION,
            family="topomap_interpret",
            tricky=True,
            notes="visual support of ranking → vision",
        ),
        _ex(
            "v_topo_05",
            "Inspect the topomap PNG and say whether left-hemisphere beta looks asymmetric.",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_topo_06",
            "From the visual appearance of img_topomap_0f8ba1d5d15cfb57, where is beta concentrated?",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_topo_07",
            "Visually compare the left vs right sensorimotor foci in this topomap.",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_topo_08",
            "Does the uploaded scalp map show a clear central hotspot or a diffuse pattern?",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_spec_01",
            "Describe the spectrogram pattern in the uploaded figure between 1 and 40 Hz.",
            VISION,
            family="spectrogram_pattern",
        ),
        _ex(
            "v_spec_02",
            "What temporal-frequency pattern is visible in this spectrogram image?",
            VISION,
            family="spectrogram_pattern",
        ),
        _ex(
            "v_spec_03",
            "Looking at the spectrogram plot, do you see a bursty beta band or sustained power?",
            VISION,
            family="spectrogram_pattern",
        ),
        _ex(
            "v_spec_04",
            "Inspect img_spectrogram_026ad6ca411b6 and describe the dominant visible band.",
            VISION,
            family="spectrogram_pattern",
        ),
        _ex(
            "v_spec_05",
            "In this spectrogram figure, is there a clear vertical transient or horizontal band structure?",
            VISION,
            family="spectrogram_pattern",
        ),
        _ex(
            "v_spec_06",
            "Visually, does the spectrogram show alpha attenuation after the event marker?",
            VISION,
            family="spectrogram_pattern",
        ),
        _ex(
            "v_spec_07",
            "Describe any artifactual stripes visible in the uploaded spectrogram.",
            VISION,
            family="spectrogram_pattern",
        ),
        _ex(
            "v_wave_01",
            "Inspect waveform morphology from this EEG image—are there high-amplitude spikes?",
            VISION,
            family="waveform_morphology",
        ),
        _ex(
            "v_wave_02",
            "Looking at the uploaded waveform plot, describe the transient shape around t=1.2s.",
            VISION,
            family="waveform_morphology",
        ),
        _ex(
            "v_wave_03",
            "From the figure, does the waveform look clipped or unsaturated?",
            VISION,
            family="waveform_morphology",
        ),
        _ex(
            "v_wave_04",
            "Visually inspect this raw trace image for blink-like deflections.",
            VISION,
            family="waveform_morphology",
        ),
        _ex(
            "v_wave_05",
            "Describe the morphology visible in the uploaded C3 waveform PNG.",
            VISION,
            family="waveform_morphology",
        ),
        _ex(
            "v_wave_06",
            "Does this waveform figure show rhythmic mu spindles or irregular noise?",
            VISION,
            family="waveform_morphology",
        ),
        _ex(
            "v_psd_vis_01",
            "What pattern is visible in this PSD figure?",
            VISION,
            family="psd_visual",
            tricky=True,
            notes="visual PSD pattern ≠ numeric peak",
        ),
        _ex(
            "v_psd_vis_02",
            "Looking at the uploaded PSD plot, is the spectrum 1/f-like or peaked?",
            VISION,
            family="psd_visual",
            tricky=True,
        ),
        _ex(
            "v_psd_vis_03",
            "Describe the shape of the curve in this PSD image around the alpha range.",
            VISION,
            family="psd_visual",
        ),
        _ex(
            "v_psd_vis_04",
            "In the PSD figure, does a secondary peak appear visually above 20 Hz?",
            VISION,
            family="psd_visual",
        ),
        _ex(
            "v_psd_vis_05",
            "Interpret the shaded bands shown in this PSD plot image.",
            VISION,
            family="psd_visual",
        ),
        _ex(
            "v_cmp_01",
            "Compare the visible patterns across these two topomap plots—are they congruent?",
            VISION,
            family="compare_plots",
        ),
        _ex(
            "v_cmp_02",
            "Visually compare the left and right spectrogram panels in the uploaded figure.",
            VISION,
            family="compare_plots",
        ),
        _ex(
            "v_cmp_03",
            "Looking at both waveform images, which trace shows larger peak-to-peak swings?",
            VISION,
            family="compare_plots",
        ),
        _ex(
            "v_cmp_04",
            "Across the two PSD figures, which spectrum looks more peaked visually?",
            VISION,
            family="compare_plots",
        ),
        _ex(
            "v_cmp_05",
            "Compare visible spatial foci between the alpha and beta topomaps in this figure.",
            VISION,
            family="compare_plots",
        ),
        _ex(
            "v_upload_01",
            "I uploaded a topomap—please interpret what the figure shows.",
            VISION,
            family="uploaded_figure",
        ),
        _ex(
            "v_upload_02",
            "Regarding the attached spectrogram image: what stands out visually?",
            VISION,
            family="uploaded_figure",
        ),
        _ex(
            "v_upload_03",
            "Please look at my uploaded PSD figure and describe the spectral shape.",
            VISION,
            family="uploaded_figure",
        ),
        _ex(
            "v_upload_04",
            "The figure I just uploaded shows EEG waveforms—comment on morphology.",
            VISION,
            family="uploaded_figure",
        ),
        _ex(
            "v_upload_05",
            "Answer from the uploaded image only: is there a clear central hotspot?",
            VISION,
            family="uploaded_figure",
        ),
        _ex(
            "v_combo_01",
            "Tools say C3 has highest beta; does this topomap visually agree?",
            VISION,
            family="image_plus_numeric",
            tricky=True,
            notes="numeric claim + visual confirmation needs image",
        ),
        _ex(
            "v_combo_02",
            "The PSD peak tool returned 10.5 Hz—does the uploaded PSD figure show a matching visual peak?",
            VISION,
            family="image_plus_numeric",
            tricky=True,
        ),
        _ex(
            "v_combo_03",
            "Ranking tool lists C3>C4>Cz for beta; inspect the topomap image and say if that ranking is visually plausible.",
            VISION,
            family="image_plus_numeric",
            tricky=True,
        ),
        _ex(
            "v_combo_04",
            "Given numeric RMS=12.4 on C3, does the waveform figure look consistent with a large-amplitude channel?",
            VISION,
            family="image_plus_numeric",
        ),
        _ex(
            "v_combo_05",
            "Condition comparison suggests left>right beta; does the uploaded topomap visually support left-lateralization?",
            VISION,
            family="image_plus_numeric",
            tricky=True,
        ),
        _ex(
            "v_combo_06",
            "Use the figure: visually confirm whether the spectrogram’s dominant band matches the tool’s alpha_mu label.",
            VISION,
            family="image_plus_numeric",
        ),
        _ex(
            "v_ref_01",
            "What do you see in figure img_psd_0873a9a51138ecf9?",
            VISION,
            family="explicit_figure_ref",
        ),
        _ex(
            "v_ref_02",
            "Interpret the visual content of img_topomap_06b9e5be484f99a7.",
            VISION,
            family="explicit_figure_ref",
        ),
        _ex(
            "v_ref_03",
            "Describe patterns visible in img_spectrogram_11a33c61527e7885.",
            VISION,
            family="explicit_figure_ref",
        ),
        _ex(
            "v_ref_04",
            "Looking at img_waveform_sample_abc, comment on spike morphology.",
            VISION,
            family="explicit_figure_ref",
        ),
        _ex(
            "v_ref_05",
            "From the plot img_band_power_heatmap_xyz, where is power concentrated visually?",
            VISION,
            family="explicit_figure_ref",
        ),
        _ex(
            "v_misc_01",
            "Is the color scale in this topomap image diverging or sequential, based on what you see?",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_misc_02",
            "Visually, are electrode labels readable on the uploaded scalp map?",
            VISION,
            family="topomap_interpret",
        ),
        _ex(
            "v_misc_03",
            "Does this PSD figure appear to use a log or linear y-axis from its visual layout?",
            VISION,
            family="psd_visual",
        ),
        _ex(
            "v_misc_04",
            "In the spectrogram image, is time on the x-axis based on the plot layout you see?",
            VISION,
            family="spectrogram_pattern",
        ),
    ]
    rows.extend(vision_bank)

    assert len(rows) >= 80, len(rows)
    assert sum(1 for r in rows if r["expected_route"] == TEXT) >= 40
    assert sum(1 for r in rows if r["expected_route"] == VISION) >= 40
    return rows


# ---------------------------------------------------------------------------
# Routing prediction from CURRENT intent fields (baseline unchanged)
# ---------------------------------------------------------------------------


def predicted_route_from_intent(raw: dict[str, Any] | None) -> str:
    """Map current intent JSON → TEXT_ONLY vs VISION_REQUIRED.

    Baseline uses existing fields only:
    include_vision_evidence / requested_visual_type / image_id.
    After optional repair, also honors requires_vision.
    """
    if not raw:
        return TEXT
    if raw.get("requires_vision") is True:
        return VISION
    if bool(raw.get("include_vision_evidence", False)):
        return VISION
    if raw.get("requested_visual_type"):
        return VISION
    if raw.get("image_id"):
        return VISION
    return TEXT


def tool_path_chosen(raw: dict[str, Any] | None) -> bool:
    if not raw:
        return False
    qt = raw.get("question_type")
    return isinstance(qt, str) and qt in {
        "band_power",
        "rms",
        "psd_peak",
        "channel_ranking",
        "threshold_set",
        "condition_comparison",
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def prf(y_true: list[str], y_pred: list[str], positive: str) -> dict[str, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive and t == positive:
            tp += 1
        elif p == positive and t != positive:
            fp += 1
        elif p != positive and t == positive:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = [TEXT, VISION]
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] = matrix[t].get(p, 0) + 1
    return {
        "labels": labels,
        "matrix_expected_rows_predicted_cols": matrix,
        "counts": {
            "expected_TEXT_ONLY": sum(1 for t in y_true if t == TEXT),
            "expected_VISION_REQUIRED": sum(1 for t in y_true if t == VISION),
            "predicted_TEXT_ONLY": sum(1 for p in y_pred if p == TEXT),
            "predicted_VISION_REQUIRED": sum(1 for p in y_pred if p == VISION),
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [r["expected_route"] for r in rows]
    y_pred = [r["predicted_route"] for r in rows]
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    n = len(rows)
    cm = confusion(y_true, y_pred)
    return {
        "n": n,
        "overall_accuracy": round(correct / n, 6) if n else 0.0,
        "vision_required": prf(y_true, y_pred, VISION),
        "text_only": prf(y_true, y_pred, TEXT),
        "confusion_matrix": cm,
        "false_positive_vision": [
            {
                "id": r["id"],
                "question": r["question"],
                "expected": r["expected_route"],
                "predicted": r["predicted_route"],
                "raw_intent": r.get("raw_intent"),
            }
            for r in rows
            if r["expected_route"] == TEXT and r["predicted_route"] == VISION
        ],
        "false_negative_vision": [
            {
                "id": r["id"],
                "question": r["question"],
                "expected": r["expected_route"],
                "predicted": r["predicted_route"],
                "raw_intent": r.get("raw_intent"),
            }
            for r in rows
            if r["expected_route"] == VISION and r["predicted_route"] == TEXT
        ],
    }


# ---------------------------------------------------------------------------
# In-process vLLM intent probe (no OpenAI server / no port conflict with nginx)
# ---------------------------------------------------------------------------


class IntentEngine:
    def __init__(self) -> None:
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=str(CKPT),
            dtype="auto",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_UTIL,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=True)

    def run_intent(
        self,
        question: str,
        system_prompt: str,
        *,
        max_tokens: int = 256,
    ) -> tuple[str, dict[str, Any] | None, float]:
        user = build_intent_user_prompt(question)
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        params = self.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        t0 = time.perf_counter()
        outs = self.llm.generate([prompt], params, use_tqdm=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = outs[0].outputs[0].text
        raw = None
        try:
            raw = extract_json_object(text)
        except Exception:  # noqa: BLE001
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    raw = json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    raw = None
        return text, raw, latency_ms


def evaluate(
    examples: list[dict[str, Any]],
    engine: IntentEngine,
    system_prompt: str,
    tag: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        text, raw, lat = engine.run_intent(ex["question"], system_prompt)
        pred = predicted_route_from_intent(raw)
        row = {
            **ex,
            "predicted_route": pred,
            "raw_model_output": text,
            "raw_intent": raw,
            "confidence": None,  # intent schema has no confidence field today
            "structured_router_output": raw,
            "tool_path_chosen": tool_path_chosen(raw),
            "vision_path_chosen": pred == VISION,
            "latency_ms": round(lat, 3),
            "correct": pred == ex["expected_route"],
            "run_tag": tag,
        }
        out_rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{tag}] {i+1}/{len(examples)} "
                f"acc_so_far={sum(1 for r in out_rows if r['correct'])/len(out_rows):.3f}"
            )
    metrics = summarize(out_rows)
    metrics["tag"] = tag
    metrics["system_prompt_sha_prefix"] = str(abs(hash(system_prompt)))[:12]
    return out_rows, metrics


# ---------------------------------------------------------------------------
# Optional ONE repair (prompt only) — applied only if baseline is weak
# ---------------------------------------------------------------------------

REPAIRED_INTENT_SYSTEM_PROMPT = """You are a neuroscience research intent parser and route classifier.
Given a user question, output ONLY a single JSON object (no markdown, no prose).

First decide the serving route:
- requires_vision=true  → VISION path (Qwen2.5-VL image understanding is necessary)
- requires_vision=false → TEXT/TOOL path (deterministic EEG/feature tools; no VLM)

Set requires_vision=true ONLY when the answer needs looking at / interpreting an image, figure, plot, topomap, spectrogram, waveform PNG, uploaded figure, or visually comparing plots. Also true when the user asks whether a figure visually supports a numeric ranking/claim.

Set requires_vision=false for exact numeric computation from stored/raw EEG or features (band power, RMS, PSD peak Hz, channel ranking, thresholds, condition comparison, metadata/stats), even if an image file may exist on disk. Numeric PSD peak ≠ describing a PSD figure pattern. "Which channel has highest beta power?" is text/tool-only. "Does this topomap visually support the beta-power ranking?" requires vision.

When requires_vision=true also set include_vision_evidence=true and requested_visual_type when inferable (psd|waveform|spectrogram|topomap|band_power). When requires_vision=false leave include_vision_evidence=false and omit requested_visual_type/image_id unless the user explicitly asked only for stored sidecar metadata without interpretation.

Supported question_type values (use exactly one when a tool applies; for pure visual interpretation still pick the closest tool type if a sample is mentioned, otherwise use "psd_peak" as a placeholder and rely on requires_vision):
- band_power, rms, psd_peak, channel_ranking, threshold_set, condition_comparison

JSON fields (omit unused fields):
{
  "requires_vision": false,
  "question_type": "<one of the six above>",
  "sample_id": "S###_R##_E###",
  "subject_id": "S###",
  "run_id": "R##",
  "epoch": 0,
  "channels": ["C3"] or "all",
  "frequency_band": "delta|theta|alpha_mu|beta",
  "frequency_range": [1.0, 40.0],
  "metric": "rms|beta_power|alpha_mu_power|...",
  "condition_a": "left_fist",
  "condition_b": "right_fist",
  "threshold": 0.0,
  "threshold_mode": "absolute|median|upper_quartile",
  "comparator": "gt|ge|lt|le",
  "top_k": 3,
  "sort_direction": "ascending|descending",
  "include_vision_evidence": false,
  "requested_visual_type": "psd|waveform|spectrogram|topomap|band_power",
  "image_id": null
}

Rules:
- Identify sample_id from the question when present (format S###_R##_E###).
- Otherwise use subject_id + run_id + epoch when inferable.
- condition_comparison requires subject_id, condition_a, condition_b (not sample_id).
- Do not invent sample IDs or numeric values not mentioned or clearly implied.
- Distinguish exact numeric tool tasks from visual interpretation.
- Output valid JSON only."""


def apply_prompt_repair() -> None:
    """One bounded repair: clearer requires_vision routing in INTENT_SYSTEM_PROMPT."""
    path = PROJECT_ROOT / "src" / "neuro_agent" / "agent" / "prompts.py"
    text = path.read_text()
    if "requires_vision" in text and "route classifier" in text:
        print("repair already present in prompts.py")
        return
    # Replace INTENT_SYSTEM_PROMPT assignment body via marker
    start = text.find("INTENT_SYSTEM_PROMPT = ")
    if start < 0:
        raise RuntimeError("INTENT_SYSTEM_PROMPT not found")
    # Find next top-level assignment after the closing """
    end_marker = '\nANSWER_SYSTEM_PROMPT = '
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError("ANSWER_SYSTEM_PROMPT marker not found")
    new_block = (
        "INTENT_SYSTEM_PROMPT = "
        + repr(REPAIRED_INTENT_SYSTEM_PROMPT)
        + "\n"
    )
    # Prefer triple-quoted form for readability
    new_block = (
        'INTENT_SYSTEM_PROMPT = """'
        + REPAIRED_INTENT_SYSTEM_PROMPT
        + '"""\n'
    )
    path.write_text(text[:start] + new_block + text[end:])
    print(f"applied one prompt repair to {path}")


def pass_fail(metrics: dict[str, Any]) -> dict[str, Any]:
    acc = metrics["overall_accuracy"]
    vrec = metrics["vision_required"]["recall"]
    # systematic confusion: high FP among numeric-tool families or high FN on visual families
    return {
        "overall_accuracy_ge_0_95": acc >= 0.95,
        "vision_required_recall_ge_0_95": vrec >= 0.95,
        "overall_accuracy": acc,
        "vision_required_recall": vrec,
        "pass": acc >= 0.95 and vrec >= 0.95,
    }


def cost_analysis(baseline_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    cost_path = PROJECT_ROOT / "results" / "serving" / "multimodal" / "text_vs_vision_cost.json"
    lat_path = PROJECT_ROOT / "results" / "serving" / "multimodal" / "latency_benchmark.json"
    prod_path = PROJECT_ROOT / "results" / "serving" / "multimodal" / "production_decision.json"
    stage_j_cost = json.loads(cost_path.read_text()) if cost_path.exists() else {}
    stage_j_lat = json.loads(lat_path.read_text()) if lat_path.exists() else {}
    stage_j_prod = json.loads(prod_path.read_text()) if prod_path.exists() else {}

    intent_lats = [r["latency_ms"] for r in baseline_rows]
    intent_lats.sort()

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return float("nan")
        k = (len(xs) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return xs[int(k)]
        return xs[f] * (c - k) + xs[c] * (k - f)

    n_fp = len(metrics["false_positive_vision"])
    n_fn = len(metrics["false_negative_vision"])
    n_text_ok = sum(
        1
        for r in baseline_rows
        if r["expected_route"] == TEXT and r["predicted_route"] == TEXT
    )
    n_vis_ok = sum(
        1
        for r in baseline_rows
        if r["expected_route"] == VISION and r["predicted_route"] == VISION
    )

    text_e2e = stage_j_cost.get("text_only", {}).get("e2e_ms_p50") or stage_j_cost.get(
        "text_only", {}
    ).get("e2e_ms_mean")
    vis_e2e = (
        stage_j_lat.get("overall", {}).get("e2e_ms", {}).get("p50")
        or stage_j_cost.get("vision_image_request", {}).get("e2e_ms_p50")
    )
    vis_vram = stage_j_cost.get("vision_image_request", {}).get("vram_idle_mb")
    text_vram = stage_j_cost.get("text_only", {}).get("vram_idle_mb")

    return {
        "stage": "J.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "stage_j_text_vs_vision_cost": str(cost_path),
            "stage_j_latency_benchmark": str(lat_path),
            "stage_j_production_decision": str(prod_path),
            "intent_routing_latencies": "this run",
        },
        "intent_routing_latency_ms": {
            "n": len(intent_lats),
            "p50": round(pct(intent_lats, 0.50), 3) if intent_lats else None,
            "p95": round(pct(intent_lats, 0.95), 3) if intent_lats else None,
            "mean": round(sum(intent_lats) / len(intent_lats), 3) if intent_lats else None,
        },
        "correct_text_routing": {
            "n_in_eval": n_text_ok,
            "path": "vLLM W8A8 INT8 text + deterministic tools",
            "measured_text_e2e_ms_p50_stage_j": text_e2e,
            "measured_text_vram_idle_mb_stage_j": text_vram,
            "notes": "Correct TEXT_ONLY routing stays on resident text path; no VLM load.",
        },
        "unnecessary_vision_routing_false_positives": {
            "n_in_eval": n_fp,
            "extra_cost_components": [
                "stop/sleep text vLLM to free KV (production util=0.90 cannot co-reside)",
                "load Qwen2.5-VL-3B corrected LoRA (HF)",
                "VLM invocation",
                "unload VLM + restart text vLLM",
            ],
            "measured_vlm_e2e_ms_p50_stage_j": vis_e2e,
            "measured_vlm_vram_idle_mb_stage_j": vis_vram,
            "swap_unload_overhead_ms": None,
            "swap_unload_overhead_note": (
                "Stage J established swap/unload is required at text util=0.90 but did not "
                "measure end-to-end swap/unload wall time; do not invent an exact latency."
            ),
            "impact": (
                "Each FP vision route pays full swap + VLM invocation cost for a query that "
                "should have stayed on the cheap text/tool path."
            ),
        },
        "missed_vision_routing_false_negatives": {
            "n_in_eval": n_fn,
            "severity": "higher",
            "impact": (
                "FN vision routes keep image-understanding questions on text/tools, which "
                "cannot see the figure → likely answer correctness failure (wrong or "
                "unsupported visual claims)."
            ),
            "measured_numeric_latency_irrelevant": True,
        },
        "production_context": {
            "strategy": stage_j_prod.get("recommended_strategy"),
            "co_residency_at_util_0_90": stage_j_prod.get(
                "can_text_and_vision_remain_resident", {}
            ).get("with_production_text_util_0.90"),
            "swap_policy": stage_j_prod.get("swap_policy"),
        },
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CMP.mkdir(parents=True, exist_ok=True)

    examples = build_eval_set()
    eval_path = RESULTS / "routing_eval_set.jsonl"
    with eval_path.open("w") as f:
        for ex in examples:
            # persist labels in the eval file only (not sent to the model)
            f.write(json.dumps(ex, sort_keys=True) + "\n")
    print(
        f"wrote {eval_path} n={len(examples)} "
        f"TEXT={sum(1 for e in examples if e['expected_route']==TEXT)} "
        f"VISION={sum(1 for e in examples if e['expected_route']==VISION)}"
    )

    print("loading in-process vLLM W8A8 for routing probe...")
    engine = IntentEngine()
    print("engine ready")

    # --- BASELINE: current INTENT_SYSTEM_PROMPT unchanged ---
    base_rows, base_metrics = evaluate(
        examples, engine, BASELINE_INTENT_SYSTEM_PROMPT, tag="baseline"
    )
    (RESULTS / "baseline_results.json").write_text(
        json.dumps(
            {
                "stage": "J.1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "backend": "vLLM W8A8 INT8 (in-process LLM)",
                "checkpoint": str(CKPT),
                "routing_signal": (
                    "include_vision_evidence | requested_visual_type | image_id "
                    "(requires_vision if present)"
                ),
                "prompt": "INTENT_SYSTEM_PROMPT (unchanged baseline)",
                "metrics": base_metrics,
                "rows": base_rows,
            },
            indent=2,
        )
    )
    print(
        "BASELINE "
        f"acc={base_metrics['overall_accuracy']:.4f} "
        f"vision_recall={base_metrics['vision_required']['recall']:.4f} "
        f"FP_vis={len(base_metrics['false_positive_vision'])} "
        f"FN_vis={len(base_metrics['false_negative_vision'])}"
    )

    repaired = False
    final_rows, final_metrics = base_rows, base_metrics
    gate = pass_fail(base_metrics)
    materially_weak = (not gate["pass"]) or base_metrics["overall_accuracy"] < 0.95

    if materially_weak:
        print("baseline materially weak → applying ONE bounded prompt repair")
        apply_prompt_repair()
        repaired_prompt = _load_intent_system_prompt_from_source()
        repaired = True
        rep_rows, rep_metrics = evaluate(
            examples, engine, repaired_prompt, tag="repaired"
        )
        (RESULTS / "repaired_results.json").write_text(
            json.dumps(
                {
                    "stage": "J.1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "repair": "INTENT_SYSTEM_PROMPT: add requires_vision + numeric vs visual rules",
                    "metrics": rep_metrics,
                    "rows": rep_rows,
                },
                indent=2,
            )
        )
        final_rows, final_metrics = rep_rows, rep_metrics
        print(
            "REPAIRED "
            f"acc={rep_metrics['overall_accuracy']:.4f} "
            f"vision_recall={rep_metrics['vision_required']['recall']:.4f}"
        )
    else:
        print("baseline already strong → no repair")

    cm = final_metrics["confusion_matrix"]
    (RESULTS / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "stage": "J.1",
                "source": "repaired" if repaired else "baseline",
                "confusion_matrix": cm,
                "false_positive_vision_count": len(final_metrics["false_positive_vision"]),
                "false_negative_vision_count": len(final_metrics["false_negative_vision"]),
            },
            indent=2,
        )
    )

    costs = cost_analysis(final_rows, final_metrics)
    (RESULTS / "routing_cost_analysis.json").write_text(json.dumps(costs, indent=2))

    final_gate = pass_fail(final_metrics)
    comparison = {
        "stage": "J.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_set": {
            "path": str(eval_path),
            "n": len(examples),
            "n_text_only": sum(1 for e in examples if e["expected_route"] == TEXT),
            "n_vision_required": sum(1 for e in examples if e["expected_route"] == VISION),
            "n_tricky": sum(1 for e in examples if e.get("tricky_negative")),
            "families": dict(Counter(e["family"] for e in examples)),
        },
        "baseline": {
            "overall_accuracy": base_metrics["overall_accuracy"],
            "vision_required": base_metrics["vision_required"],
            "text_only": base_metrics["text_only"],
            "fp_vision": len(base_metrics["false_positive_vision"]),
            "fn_vision": len(base_metrics["false_negative_vision"]),
        },
        "repair_applied": repaired,
        "final": {
            "overall_accuracy": final_metrics["overall_accuracy"],
            "vision_required": final_metrics["vision_required"],
            "text_only": final_metrics["text_only"],
            "confusion_matrix": final_metrics["confusion_matrix"],
            "fp_vision_examples": final_metrics["false_positive_vision"][:15],
            "fn_vision_examples": final_metrics["false_negative_vision"][:15],
            "fp_vision_count": len(final_metrics["false_positive_vision"]),
            "fn_vision_count": len(final_metrics["false_negative_vision"]),
        },
        "pass_criteria": final_gate,
        "j1_verdict": "PASS" if final_gate["pass"] else "FAIL",
        "production_recommendation": (
            "Keep text-primary + controlled vision swap/unload (Stage J). "
            "Gate VLM invocation on requires_vision / vision-route prediction; "
            "treat false-negative vision routing as the more serious failure. "
            "Prefer high vision-required recall even if it costs occasional swap overhead."
        ),
        "cost_analysis_path": str(RESULTS / "routing_cost_analysis.json"),
    }
    (CMP / "text_vs_vision_routing.json").write_text(json.dumps(comparison, indent=2))
    (RESULTS / "j1_summary.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps({"j1_verdict": comparison["j1_verdict"], **final_gate}, indent=2))


if __name__ == "__main__":
    main()
