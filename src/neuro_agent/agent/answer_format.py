"""Deterministic grounded answer formatting from evidence bundles."""

from __future__ import annotations

from typing import Any

from neuro_agent.tools.evidence import EvidenceBundle


def format_grounded_answer(question: str, bundle: EvidenceBundle) -> str:
    """Build a sectioned answer directly from evidence (no invented numerics)."""
    tool_names = ", ".join(inv.name for inv in bundle.tool_invocations) or "none"
    units = bundle.units or "unspecified"
    warnings = bundle.warnings or bundle.uncertainty_notes or []
    uncertainty = "; ".join(warnings) if warnings else "None"

    answer_line = _answer_line(bundle)
    evidence_line = _evidence_line(bundle, units)

    return "\n".join(
        [
            f"Answer: {answer_line}",
            f"Evidence: {evidence_line}",
            f"Tools used: {tool_names}",
            f"Uncertainty: {uncertainty}",
        ]
    )


def _answer_line(bundle: EvidenceBundle) -> str:
    qtype = bundle.question_type
    num = bundle.numeric_evidence or {}

    if qtype == "band_power":
        ch = num.get("channel", "channel")
        val = num.get("value")
        band = num.get("band", "band")
        return f"{band} band power at {ch} is {val} {bundle.units or ''}".strip()

    if qtype == "rms":
        if "channel" in num:
            return f"RMS at {num['channel']} is {num['value']} {bundle.units or ''}".strip()
        if "highest_rms_channel" in num:
            return f"Highest RMS channel is {num['highest_rms_channel']}"
        return f"RMS values: {num.get('values', {})}"

    if qtype == "psd_peak":
        peak = num.get("peak_frequency_hz")
        ch = num.get("channel", "")
        return f"PSD peak frequency is {peak} Hz at channel {ch}".strip()

    if qtype == "channel_ranking" and bundle.ranked_evidence:
        ranking = bundle.ranked_evidence.get("ranking", [])
        metric = bundle.ranked_evidence.get("metric", "metric")
        if ranking:
            return f"Top channel by {metric} is {ranking[0]}"
        return "No channel ranking available"

    if qtype == "threshold_set" and bundle.set_evidence:
        channels = bundle.set_evidence.get("channels", [])
        return f"Selected channels: {', '.join(channels) if channels else 'none'}"

    if qtype == "condition_comparison" and bundle.condition_evidence:
        ce = bundle.condition_evidence
        return (
            f"Comparison of {ce.get('condition_a')} vs {ce.get('condition_b')}: "
            f"higher metric is {ce.get('higher_condition')}"
        )

    return f"See evidence for sample {bundle.metadata.get('sample_id') if bundle.metadata else 'unknown'}"


def _evidence_line(bundle: EvidenceBundle, units: str) -> str:
    qtype = bundle.question_type
    num = bundle.numeric_evidence or {}

    if qtype == "band_power":
        return (
            f"channel={num.get('channel')}, band={num.get('band')}, "
            f"value={num.get('value')} {units}"
        )

    if qtype == "rms":
        if "values" in num:
            parts = [f"{ch}={val} {units}" for ch, val in num["values"].items()]
            return "; ".join(parts)
        return f"channel={num.get('channel')}, value={num.get('value')} {units}"

    if qtype == "psd_peak":
        return (
            f"peak_frequency_hz={num.get('peak_frequency_hz')} Hz, "
            f"channel={num.get('channel')}"
        )

    if qtype == "channel_ranking" and bundle.ranked_evidence:
        values = bundle.ranked_evidence.get("values", {})
        ranking = bundle.ranked_evidence.get("ranking", [])
        parts = [f"{ch}={values.get(ch)} {units}" for ch in ranking[:5]]
        return "; ".join(parts)

    if qtype == "threshold_set" and bundle.set_evidence:
        se = bundle.set_evidence
        return (
            f"threshold={se.get('threshold_used')} {units}, "
            f"channels={se.get('channels', [])}"
        )

    if qtype == "condition_comparison" and bundle.condition_evidence:
        ce = bundle.condition_evidence
        return (
            f"{ce.get('condition_a')}={ce.get('value_a')}, "
            f"{ce.get('condition_b')}={ce.get('value_b')} {units}"
        )

    return str(num)
