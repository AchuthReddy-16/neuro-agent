"""Deterministic verifiers for neuroscience evaluation tasks."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any


REFUSAL_PATTERNS = (
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bunable to\b",
    r"\bnot enough information\b",
    r"\binsufficient\b",
)


@dataclass
class VerificationResult:
    """Outcome of verifying a model response against an example."""

    passed: bool
    verification_type: str
    parsed_answer: Any = None
    expected: Any = None
    reason: str = ""
    parse_error: bool = False
    empty_or_refusal: bool = False
    grounded_in_context: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_token(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def is_empty_or_refusal(response: str) -> bool:
    stripped = response.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    return any(re.search(p, lower) for p in REFUSAL_PATTERNS)


def _extract_numbers(text: str) -> list[float]:
    matches = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    values: list[float] = []
    for m in matches:
        try:
            values.append(float(m))
        except ValueError:
            continue
    return values


def parse_numeric(response: str) -> float | None:
    values = _extract_numbers(response)
    if not values:
        return None
    return values[0]


KNOWN_CATEGORICAL_LABELS = {
    "band_power",
    "baseline",
    "baseline_rest",
    "both_feet",
    "both_fists",
    "execution",
    "execution_both_feet",
    "execution_both_fists",
    "execution_left_fist",
    "execution_rest",
    "execution_right_fist",
    "imagery",
    "imagery_both_feet",
    "imagery_both_fists",
    "imagery_left_fist",
    "imagery_rest",
    "imagery_right_fist",
    "left_fist",
    "rest",
    "right_fist",
    "rms",
}


def _find_embedded_label(text: str, valid_labels: set[str]) -> str | None:
    normalized = normalize_token(text)
    matches: list[str] = []
    for label in valid_labels:
        label_norm = normalize_token(label)
        if label_norm and label_norm in normalized:
            matches.append(label_norm)
    if not matches:
        return None
    return sorted(matches, key=len, reverse=True)[0]


def parse_categorical(response: str, valid_labels: set[str] | None = None) -> str | None:
    text = response.strip()
    if not text:
        return None

    labels = valid_labels or KNOWN_CATEGORICAL_LABELS

    patterns = (
        r"(?:answer|label|result|condition|sample)\s*(?:is|:)\s*([A-Za-z0-9_ -]+)",
        r"\bis\s+([A-Za-z0-9_ -]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidate = normalize_token(match.group(1).strip(" .\"'`"))
            if candidate in {normalize_token(x) for x in labels}:
                return candidate
            embedded = _find_embedded_label(match.group(1), labels)
            if embedded:
                return embedded

    first_line = text.splitlines()[0].strip()
    first_line = re.sub(r"^(answer|label|result)\s*:\s*", "", first_line, flags=re.I)
    first_line = first_line.strip("\"'` ")
    candidate = normalize_token(first_line) if first_line else normalize_token(text)
    label_norms = {normalize_token(x) for x in labels}
    if candidate in label_norms:
        return candidate

    embedded = _find_embedded_label(text, labels)
    if embedded:
        return embedded
    return candidate if candidate else None


def _extract_channel_dict(context: dict[str, Any]) -> dict[str, Any]:
    """Return the first dict-valued context field that looks like channel metrics."""
    values = context.get("values")
    if isinstance(values, dict) and values:
        return values
    for val in context.values():
        if isinstance(val, dict) and val:
            sample_keys = list(val.keys())[:3]
            if sample_keys and all(isinstance(k, str) and re.match(r"^[A-Z]", k) for k in sample_keys):
                return val
    return {}


def parse_set_membership(response: str, known_channels: set[str] | None = None) -> set[str]:
    """Parse a set of channel labels from a free-form response."""
    ranked = parse_ranking(response, known_channels=known_channels)
    return {item.upper() for item in ranked}


def parse_ranking(response: str, known_channels: set[str] | None = None) -> list[str]:
    text = response.upper()
    found: list[str] = []
    if known_channels:
        for ch in sorted(known_channels, key=len, reverse=True):
            if re.search(rf"\b{re.escape(ch.upper())}\b", text):
                found.append(ch.upper())
    else:
        for m in re.findall(r"\b[A-Z][A-Z0-9]{1,3}\b", text):
            if m not in found:
                found.append(m)
    return found


def _numeric_close(pred: float, target: float, tolerance: dict[str, float] | None) -> bool:
    tol = tolerance or {"absolute": 1e-6, "relative": 1e-5}
    abs_tol = float(tol.get("absolute", 1e-6))
    rel_tol = float(tol.get("relative", 1e-5))
    if math.isclose(pred, target, rel_tol=0.0, abs_tol=abs_tol):
        return True
    scale = max(abs(target), 1.0)
    return abs(pred - target) <= max(abs_tol, rel_tol * scale)


def _flatten_numeric_values(obj: Any) -> list[float]:
    values: list[float] = []
    if isinstance(obj, bool):
        return values
    if isinstance(obj, (int, float)):
        values.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            values.extend(_flatten_numeric_values(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            values.extend(_flatten_numeric_values(v))
    return values


def _value_grounded_in_context(value: float, context: dict[str, Any], tolerance: dict[str, float] | None) -> bool:
    for ctx_val in _flatten_numeric_values(context):
        if _numeric_close(value, ctx_val, tolerance):
            return True
    return False


def verify_example(example: dict[str, Any], response: str) -> VerificationResult:
    """Verify a model response for one evaluation example."""
    vtype = example["verification_type"]
    expected = example["ground_truth"]
    tolerance = example.get("tolerance")

    if is_empty_or_refusal(response):
        return VerificationResult(
            passed=False,
            verification_type=vtype,
            expected=expected,
            reason="empty_or_refusal",
            empty_or_refusal=True,
        )

    if vtype == "numeric":
        parsed = parse_numeric(response)
        if parsed is None:
            return VerificationResult(
                passed=False,
                verification_type=vtype,
                expected=expected,
                reason="unparseable_numeric",
                parse_error=True,
            )
        passed = _numeric_close(parsed, float(expected), tolerance)
        grounded = _value_grounded_in_context(parsed, example.get("context", {}), tolerance)
        return VerificationResult(
            passed=passed,
            verification_type=vtype,
            parsed_answer=parsed,
            expected=expected,
            reason="match" if passed else "numeric_mismatch",
            grounded_in_context=grounded,
        )

    if vtype == "categorical":
        parsed = parse_categorical(response)
        if parsed is None:
            return VerificationResult(
                passed=False,
                verification_type=vtype,
                expected=expected,
                reason="unparseable_categorical",
                parse_error=True,
            )
        expected_norm = normalize_token(str(expected))
        passed = parsed == expected_norm
        return VerificationResult(
            passed=passed,
            verification_type=vtype,
            parsed_answer=parsed,
            expected=expected_norm,
            reason="match" if passed else "categorical_mismatch",
        )

    if vtype == "set":
        channel_values = _extract_channel_dict(example.get("context", {}))
        known = set(channel_values.keys()) if channel_values else set()
        parsed_set = parse_set_membership(response, known_channels=known)
        if not parsed_set:
            return VerificationResult(
                passed=False,
                verification_type=vtype,
                expected=expected,
                reason="unparseable_set",
                parse_error=True,
            )
        expected_set = {str(x).upper() for x in (expected if isinstance(expected, list) else [expected])}
        passed = parsed_set == expected_set
        return VerificationResult(
            passed=passed,
            verification_type=vtype,
            parsed_answer=sorted(parsed_set),
            expected=sorted(expected_set),
            reason="match" if passed else "set_mismatch",
        )

    if vtype == "ranking":
        channel_values = _extract_channel_dict(example.get("context", {}))
        known = set(channel_values.keys()) if channel_values else set()
        parsed = parse_ranking(response, known_channels=known)
        if not parsed:
            return VerificationResult(
                passed=False,
                verification_type=vtype,
                expected=expected,
                reason="unparseable_ranking",
                parse_error=True,
            )
        expected_list = [str(x).upper() for x in (expected if isinstance(expected, list) else [expected])]
        passed = parsed[0] in expected_list
        return VerificationResult(
            passed=passed,
            verification_type=vtype,
            parsed_answer=parsed,
            expected=expected_list,
            reason="match" if passed else "ranking_mismatch",
        )

    return VerificationResult(
        passed=False,
        verification_type=vtype,
        expected=expected,
        reason=f"unsupported_verification_type:{vtype}",
        parse_error=True,
    )


def format_eval_prompt(example: dict[str, Any], system_prompt: str) -> str:
    """Build a deterministic evaluation prompt."""
    context_json = json.dumps(example.get("context", {}), indent=2, sort_keys=True)
    return (
        f"{system_prompt.strip()}\n\n"
        f"Context:\n{context_json}\n\n"
        f"Question: {example['question'].strip()}\n\n"
        f"Answer:"
    )
