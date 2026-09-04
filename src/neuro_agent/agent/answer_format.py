"""Deterministic grounded answer formatting from evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuro_agent.tools.evidence import EvidenceBundle


@dataclass
class GroundedAnswerParts:
    """User-facing parts derived only from the evidence bundle."""

    answer: str
    evidence_summary: str
    tools: str
    uncertainty: str


def format_grounded_parts(question: str, bundle: EvidenceBundle) -> GroundedAnswerParts:
    """Build natural-language answer parts from evidence (no invented numerics)."""
    tool_names = ", ".join(inv.name for inv in bundle.tool_invocations) or "none"
    units = bundle.units or ""
    warnings = bundle.warnings or bundle.uncertainty_notes or []
    uncertainty = "; ".join(warnings) if warnings else "None"
    answer = _natural_answer(question, bundle, units)
    evidence = _evidence_line(bundle, units or "unspecified")
    return GroundedAnswerParts(
        answer=answer,
        evidence_summary=evidence,
        tools=tool_names,
        uncertainty=uncertainty,
    )


def format_grounded_answer(question: str, bundle: EvidenceBundle) -> str:
    """Compatibility wrapper: natural answer plus optional sectioned metadata for parsers."""
    parts = format_grounded_parts(question, bundle)
    # Prefer a clean user-facing answer. Keep light section markers only when
    # uncertainty is non-trivial so legacy extractors still work.
    if parts.uncertainty and parts.uncertainty != "None":
        return f"{parts.answer}\nUncertainty: {parts.uncertainty}"
    return parts.answer


def _display_units(units: str | None) -> str:
    if not units:
        return ""
    u = units.strip()
    if u.lower() in {"uv2", "uv²", "uV2"}:
        return "μV²"
    return u


def _fmt_num(val: Any) -> str:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(x) >= 100:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.4g}"


def _natural_answer(question: str, bundle: EvidenceBundle, units: str) -> str:
    qtype = bundle.question_type
    num = bundle.numeric_evidence or {}
    unit = _display_units(units or bundle.units)

    if qtype == "band_power":
        ch = num.get("channel", "the selected channel")
        val = num.get("value")
        band = num.get("band", "band")
        return (
            f"{str(band).capitalize()} band power at {ch} is "
            f"{_fmt_num(val)}{(' ' + unit) if unit else ''}."
        ).strip()

    if qtype == "rms":
        if "channel" in num:
            return (
                f"RMS at {num['channel']} is {_fmt_num(num['value'])}"
                f"{(' ' + unit) if unit else ''}."
            ).strip()
        if "highest_rms_channel" in num:
            return f"The highest RMS channel is {num['highest_rms_channel']}."
        return f"RMS values: {num.get('values', {})}"

    if qtype == "psd_peak":
        peak = num.get("peak_frequency_hz")
        ch = num.get("channel", "")
        ch_bit = f" at channel {ch}" if ch else ""
        return f"The PSD peak frequency{ch_bit} is {_fmt_num(peak)} Hz."

    if qtype == "channel_ranking" and bundle.ranked_evidence:
        return _ranking_answer(question, bundle, unit)

    if qtype == "threshold_set" and bundle.set_evidence:
        channels = bundle.set_evidence.get("channels", [])
        if not channels:
            return "No channels met the selected threshold."
        return f"Selected channels above threshold: {', '.join(str(c) for c in channels)}."

    if qtype == "condition_comparison" and bundle.condition_evidence:
        ce = bundle.condition_evidence
        a = ce.get("condition_a", "condition A")
        b = ce.get("condition_b", "condition B")
        higher = ce.get("higher_condition") or ce.get("winner")
        va = ce.get("value_a")
        vb = ce.get("value_b")
        if va is not None and vb is not None and higher:
            return (
                f"Comparing {a} vs {b}, {higher} is higher "
                f"({a}={_fmt_num(va)}, {b}={_fmt_num(vb)}"
                f"{(' ' + unit) if unit else ''})."
            )
        return f"Comparison of {a} vs {b}: higher metric is {higher}."

    sample = bundle.metadata.get("sample_id") if bundle.metadata else None
    if sample:
        return f"See computed evidence for sample {sample}."
    return "See computed evidence for details."


def _ranking_answer(question: str, bundle: EvidenceBundle, unit: str) -> str:
    ranked = bundle.ranked_evidence or {}
    ranking = list(ranked.get("ranking") or [])
    values = ranked.get("values") or {}
    metric = str(ranked.get("metric") or "metric").replace("_", " ")
    top_k = ranked.get("top_k")
    try:
        k = int(top_k) if top_k is not None else None
    except (TypeError, ValueError):
        k = None
    # Infer N from the question when present ("five", "top 5", "5 channels").
    q = (question or "").lower()
    for word, n in (
        ("five", 5),
        ("four", 4),
        ("three", 3),
        ("two", 2),
        ("ten", 10),
    ):
        if word in q:
            k = n
            break
    m = None
    import re

    m = re.search(r"\btop\s*(\d+)\b|\b(\d+)\s+channels?\b|\bwhich\s+(\d+)\b", q)
    if m:
        k = int(next(g for g in m.groups() if g))
    if k is None:
        k = min(5, len(ranking)) if ranking else 0
    ranking = ranking[:k]
    if not ranking:
        return "No channel ranking is available for this sample."

    unit_bit = f" {unit}" if unit else ""
    if len(ranking) == 1:
        ch = ranking[0]
        if ch in values:
            return (
                f"The highest {metric} channel is {ch} "
                f"at {_fmt_num(values[ch])}{unit_bit}."
            )
        return f"The highest {metric} channel is {ch}."

    names = ", ".join(str(c) for c in ranking[:-1]) + f", and {ranking[-1]}"
    detail_parts = []
    for i, ch in enumerate(ranking):
        if ch not in values:
            continue
        if i == 0:
            detail_parts.append(
                f"{ch} is highest at {_fmt_num(values[ch])}{unit_bit}"
            )
        else:
            detail_parts.append(f"{ch} at {_fmt_num(values[ch])}{unit_bit}")
    detail = "; ".join(detail_parts) if detail_parts else ""
    count_word = {
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        10: "ten",
    }.get(len(ranking), str(len(ranking)))
    head = f"The {count_word} channels with the highest {metric} are {names}."
    if detail:
        # Prefer flowing prose for the common top-5 case
        if len(ranking) >= 3 and all(ch in values for ch in ranking):
            seq = []
            for i, ch in enumerate(ranking):
                if i == 0:
                    seq.append(
                        f"{ch} is highest at {_fmt_num(values[ch])}{unit_bit}"
                    )
                else:
                    seq.append(f"{ch} at {_fmt_num(values[ch])}{unit_bit}")
            # "T8 is highest at X, followed by IZ at Y, O2 at Z, ..."
            followed = seq[0]
            if len(seq) > 1:
                followed += ", followed by " + ", ".join(seq[1:])
            return f"{head} {followed}."
        return f"{head} {detail}."
    return head


def _evidence_line(bundle: EvidenceBundle, units: str) -> str:
    qtype = bundle.question_type
    num = bundle.numeric_evidence or {}
    unit = _display_units(units)

    if qtype == "band_power":
        return (
            f"channel={num.get('channel')}, band={num.get('band')}, "
            f"value={num.get('value')} {unit}"
        ).strip()

    if qtype == "rms":
        if "values" in num:
            parts = [f"{ch}={val} {unit}" for ch, val in num["values"].items()]
            return "; ".join(parts)
        return f"channel={num.get('channel')}, value={num.get('value')} {unit}".strip()

    if qtype == "psd_peak":
        return (
            f"peak_frequency_hz={num.get('peak_frequency_hz')} Hz, "
            f"channel={num.get('channel')}"
        )

    if qtype == "channel_ranking" and bundle.ranked_evidence:
        values = bundle.ranked_evidence.get("values", {})
        ranking = bundle.ranked_evidence.get("ranking", [])
        parts = [f"{ch}={values.get(ch)} {unit}" for ch in ranking[:5]]
        return "; ".join(parts).strip()

    if qtype == "threshold_set" and bundle.set_evidence:
        se = bundle.set_evidence
        return (
            f"threshold={se.get('threshold_used')} {unit}, "
            f"channels={se.get('channels', [])}"
        ).strip()

    if qtype == "condition_comparison" and bundle.condition_evidence:
        ce = bundle.condition_evidence
        return (
            f"{ce.get('condition_a')}={ce.get('value_a')}, "
            f"{ce.get('condition_b')}={ce.get('value_b')} {unit}"
        ).strip()

    return str(num)
