"""NL → structured intent parsing and validation for research routing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from neuro_agent.agent.prompts import SUPPORTED_INTENTS
from neuro_agent.tools.evidence import ResearchToolRequest
from neuro_agent.tools.schemas import SAMPLE_ID_RE

SAMPLE_ID_PATTERN = re.compile(r"\b(S\d{3}_R\d{2}_E\d{3})\b")
PARTIAL_SAMPLE_ID_PATTERN = re.compile(r"^(S\d{3})_(R\d{2})_E(\d{1,3})$")
SUBJECT_PATTERN = re.compile(r"\b(S\d{3})\b")
RUN_PATTERN = re.compile(r"\b(R\d{2})\b")
EPOCH_PATTERN = re.compile(r"\b(?:epoch|E)\s*(\d{1,3})\b", re.IGNORECASE)


class IntentValidationError(ValueError):
    """Raised when parsed intent JSON fails schema validation."""


@dataclass
class IntentParseResult:
    """Outcome of intent parsing and validation."""

    success: bool
    request: ResearchToolRequest | None = None
    raw_json: dict[str, Any] | None = None
    question_type: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""
    text = text.strip()
    if not text:
        raise IntentValidationError("Empty model output")

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fenced code block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))

    # First balanced brace object
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

    raise IntentValidationError(f"No JSON object found in: {text[:200]!r}")


def _normalize_channels(value: Any) -> list[str] | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [str(ch).upper() for ch in value]
    raise IntentValidationError(f"Invalid channels type: {type(value)}")


def _normalize_frequency_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise IntentValidationError(f"Invalid frequency_range: {value!r}")


def _has_sample_locator(data: dict[str, Any]) -> bool:
    if data.get("sample_id"):
        return True
    return bool(data.get("subject_id") and data.get("run_id") and data.get("epoch") is not None)


def _normalize_sample_id(sample_id: str | None, data: dict[str, Any]) -> str | None:
    """Normalize truncated sample IDs or build from subject/run/epoch."""
    if sample_id:
        sample_id = str(sample_id).strip().upper()
        if SAMPLE_ID_RE.match(sample_id):
            return sample_id
        partial = PARTIAL_SAMPLE_ID_PATTERN.match(sample_id)
        if partial:
            subject, run_id, epoch_str = partial.groups()
            return f"{subject}_{run_id}_E{int(epoch_str):03d}"

    subject_id = data.get("subject_id")
    run_id = data.get("run_id")
    epoch = data.get("epoch")
    if subject_id and run_id and epoch is not None:
        return f"{str(subject_id).strip().upper()}_{str(run_id).strip().upper()}_E{int(epoch):03d}"
    return sample_id if sample_id and SAMPLE_ID_RE.match(sample_id) else None


def validate_intent(data: dict[str, Any]) -> ResearchToolRequest:
    """Validate parsed JSON and build a ResearchToolRequest."""
    qtype = data.get("question_type")
    if not qtype or not isinstance(qtype, str):
        raise IntentValidationError("question_type is required")
    qtype = qtype.strip()
    if qtype not in SUPPORTED_INTENTS:
        raise IntentValidationError(
            f"Unsupported question_type: {qtype!r}. "
            f"Expected one of {list(SUPPORTED_INTENTS)}"
        )

    sample_id = _normalize_sample_id(
        str(data["sample_id"]).strip().upper() if data.get("sample_id") else None,
        data,
    )
    if sample_id and not SAMPLE_ID_RE.match(sample_id):
        raise IntentValidationError(f"Invalid sample_id format: {sample_id!r}")

    subject_id = data.get("subject_id")
    if subject_id is not None:
        subject_id = str(subject_id).strip().upper()

    run_id = data.get("run_id")
    if run_id is not None:
        run_id = str(run_id).strip().upper()

    epoch = data.get("epoch")
    if epoch is not None:
        epoch = int(epoch)

    if qtype == "condition_comparison":
        if not subject_id:
            raise IntentValidationError("condition_comparison requires subject_id")
        if not data.get("condition_a") or not data.get("condition_b"):
            raise IntentValidationError("condition_comparison requires condition_a and condition_b")
    elif not _has_sample_locator(data):
        raise IntentValidationError(
            f"{qtype} requires sample_id or (subject_id, run_id, epoch)"
        )

    threshold_mode = data.get("threshold_mode", "absolute")
    if threshold_mode not in ("absolute", "median", "upper_quartile"):
        raise IntentValidationError(f"Invalid threshold_mode: {threshold_mode!r}")

    comparator = data.get("comparator")
    if comparator is not None and comparator not in ("gt", "ge", "lt", "le"):
        raise IntentValidationError(f"Invalid comparator: {comparator!r}")

    sort_direction = data.get("sort_direction", "descending")
    if sort_direction not in ("ascending", "descending"):
        raise IntentValidationError(f"Invalid sort_direction: {sort_direction!r}")

    top_k = data.get("top_k")
    if top_k is not None:
        top_k = int(top_k)
        if top_k < 1:
            raise IntentValidationError("top_k must be >= 1")

    return ResearchToolRequest(
        question_type=qtype,  # type: ignore[arg-type]
        sample_id=sample_id,
        subject_id=subject_id,
        run_id=run_id,
        epoch=epoch,
        channels=_normalize_channels(data.get("channels")),
        frequency_band=data.get("frequency_band"),
        frequency_range=_normalize_frequency_range(data.get("frequency_range")),
        metric=data.get("metric"),
        condition_a=data.get("condition_a"),
        condition_b=data.get("condition_b"),
        threshold=float(data["threshold"]) if data.get("threshold") is not None else None,
        threshold_mode=threshold_mode,
        comparator=comparator,
        top_k=top_k,
        sort_direction=sort_direction,
        requested_visual_type=data.get("requested_visual_type"),
        image_id=data.get("image_id"),
        include_vision_evidence=bool(data.get("include_vision_evidence", False)),
    )


def parse_and_validate_intent(model_output: str) -> IntentParseResult:
    """Parse model JSON output and validate into ResearchToolRequest."""
    try:
        raw = extract_json_object(model_output)
        request = validate_intent(raw)
        return IntentParseResult(
            success=True,
            request=request,
            raw_json=raw,
            question_type=request.question_type,
        )
    except (IntentValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return IntentParseResult(success=False, error=str(exc))


def intent_matches_expected(
    parsed: IntentParseResult,
    expected: dict[str, Any],
) -> bool:
    """Check whether parsed intent matches expected question_type and key fields."""
    if not parsed.success or parsed.request is None:
        return False
    if parsed.request.question_type != expected.get("question_type"):
        return False
    for key, value in expected.items():
        if key == "question_type":
            continue
        actual = getattr(parsed.request, key, None)
        if actual is None:
            continue
        if key == "channels" and isinstance(value, list) and isinstance(actual, list):
            if [c.upper() for c in actual] != [c.upper() for c in value]:
                return False
        elif key == "epoch":
            if int(actual) != int(value):
                return False
        elif str(actual).lower() != str(value).lower():
            return False
    return True
