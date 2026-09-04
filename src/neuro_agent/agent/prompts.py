"""Prompt templates for intent selection and grounded answer generation."""

from __future__ import annotations

import json
from typing import Any

SUPPORTED_INTENTS = (
    "band_power",
    "rms",
    "psd_peak",
    "channel_ranking",
    "threshold_set",
    "condition_comparison",
)

INTENT_SYSTEM_PROMPT = """You are a neuroscience research intent parser and route classifier.
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

ANSWER_SYSTEM_PROMPT = """You are a neuroscience research assistant.
You must answer only from the supplied evidence. Do not invent numeric values.

Write a concise natural-language research answer that directly addresses the user's question.
- Use clear scientific prose (not telegraphic labels).
- Include the important numeric values with units when present in the evidence.
- If the user asked for N items (e.g. five channels), list all N — do not stop at the top item.
- Do NOT prefix the answer with labels like "Answer:", "Evidence:", or "Tools used:".
- End with a single line exactly of the form: Uncertainty: <brief limitation or None>"""


CONVERSATIONAL_SYSTEM_PROMPT = """You are a helpful neuroscience research assistant.
Answer the user's message naturally and concisely.
- For greetings, respond warmly and briefly offer how you can help with EEG/neuroscience research.
- For conceptual questions, explain clearly without inventing sample-specific numbers.
- For follow-ups, use any supplied prior conversation/result context; do not invent new numeric values.
- Do NOT invent tool results, channel rankings, or experimental measurements.
- Do NOT prefix with labels like "Answer:", "Evidence:", or "Tools used:".
- If a brief limitation is needed, end with: Uncertainty: <brief note or None>
Otherwise end with: Uncertainty: None"""


def build_intent_user_prompt(question: str) -> str:
    return f"Question: {question.strip()}\n\nJSON:"


def build_conversational_user_prompt(
    question: str,
    *,
    prior_context: str | None = None,
    history_snippet: str | None = None,
) -> str:
    parts = []
    if history_snippet:
        parts.append(f"Recent conversation:\n{history_snippet.strip()}")
    if prior_context:
        parts.append(f"Relevant prior result:\n{prior_context.strip()}")
    parts.append(f"Current user message:\n{question.strip()}")
    parts.append("Assistant reply:")
    return "\n\n".join(parts)


RECOVERY_ANSWER_SYSTEM_PROMPT = ANSWER_SYSTEM_PROMPT + (
    "\n\nRecovery mode: correct any unsupported numeric claims. "
    "Use ONLY values present in the evidence bundle."
)


def build_answer_user_prompt(
    question: str,
    evidence: dict[str, Any],
) -> str:
    payload = {
        "question": question.strip(),
        "metadata": evidence.get("metadata"),
        "numeric_evidence": evidence.get("numeric_evidence"),
        "ranked_evidence": evidence.get("ranked_evidence"),
        "set_evidence": evidence.get("set_evidence"),
        "condition_evidence": evidence.get("condition_evidence"),
        "vision_evidence": evidence.get("vision_evidence"),
        "provenance": evidence.get("provenance"),
        "warnings": evidence.get("warnings"),
        "uncertainty_notes": evidence.get("uncertainty_notes"),
        "units": evidence.get("units"),
        "tool_invocations": [
            {"name": inv["name"], "success": inv.get("success", True)}
            for inv in evidence.get("tool_invocations", [])
        ],
    }
    return (
        "Answer using ONLY the evidence below. Do not invent numeric values.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )
