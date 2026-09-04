"""Verifiable reward functions for RLVR training."""

from __future__ import annotations

import json
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from neuro_agent.evaluation.verifiers import (
    _extract_channel_dict,
    _numeric_close,
    is_empty_or_refusal,
    normalize_token,
    parse_categorical,
    parse_numeric,
    parse_ranking,
    parse_set_membership,
    verify_example,
)

# All multimodal RLVR rewards are bounded in [0.0, 1.0].
REWARD_MIN = 0.0
REWARD_MAX = 1.0

REWARD_SPEC = {
    "categorical": "1.0 on exact normalized label match, else 0.0",
    "numeric": "1.0 within tolerance; linear partial credit to 0 at 3x tolerance band",
    "ranking": "1.0 exact order; partial = 0.7 * position accuracy + 0.3 * set overlap",
    "set": "F1 over predicted vs expected channel sets; 1.0 on exact match",
    "waveform_numeric": "numeric reward with waveform RMS tolerance shaping",
    "spectrogram_peak_frequency": "numeric reward with peak-frequency tolerance shaping",
}


@dataclass
class RewardTracker:
    """Thread-safe accumulator for per-task reward statistics during RLVR."""

    totals: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, task_family: str, reward: float) -> None:
        with self._lock:
            self.totals[task_family].append(reward)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            for task, values in sorted(self.totals.items()):
                if not values:
                    continue
                out[task] = {
                    "count": float(len(values)),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        return out


# Global tracker used by multimodal_verifiable_reward during training runs.
MULTIMODAL_REWARD_TRACKER = RewardTracker()


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for message in completion:
            if isinstance(message, dict) and message.get("role") == "assistant":
                parts.append(str(message.get("content", "")))
        if parts:
            return "\n".join(parts).strip()
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", "")).strip()
    if isinstance(completion, dict):
        return str(completion.get("content", completion)).strip()
    return str(completion).strip()


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _tolerance_band(target: float, tolerance: dict[str, float] | None) -> float:
    tol = tolerance or {"absolute": 1e-6, "relative": 1e-5}
    abs_tol = float(tol.get("absolute", 1e-6))
    rel_tol = float(tol.get("relative", 1e-5))
    scale = max(abs(target), 1.0)
    return max(abs_tol, rel_tol * scale)


def numeric_graded_reward(
    parsed: float | None,
    target: float,
    tolerance: dict[str, float] | None,
    *,
    shaping: str = "default",
) -> float:
    """Tolerance-shaped numeric reward in [0, 1]."""
    if parsed is None:
        return REWARD_MIN
    if _numeric_close(parsed, float(target), tolerance):
        return REWARD_MAX

    band = _tolerance_band(float(target), tolerance)
    if shaping == "waveform_rms":
        band = max(band, 0.05 * max(abs(float(target)), 1e-6))
    elif shaping == "spectrogram_peak":
        band = max(band, 0.5)

    err = abs(parsed - float(target))
    outer = band * 3.0
    if err >= outer:
        return REWARD_MIN
    return max(REWARD_MIN, REWARD_MAX - (err - band) / max(outer - band, 1e-12))


def ranking_graded_reward(parsed: list[str], expected: list[Any]) -> float:
    """Exact order = 1.0; partial credit for position accuracy and overlap."""
    if not parsed:
        return REWARD_MIN
    exp = [str(x).upper() for x in expected]
    pred = [str(x).upper() for x in parsed]
    if pred == exp:
        return REWARD_MAX
    if not exp:
        return REWARD_MIN
    n = len(exp)
    position_score = sum(1.0 for i, item in enumerate(exp) if i < len(pred) and pred[i] == item) / n
    overlap = len(set(pred) & set(exp)) / n
    return max(REWARD_MIN, min(REWARD_MAX, 0.7 * position_score + 0.3 * overlap))


def set_graded_reward(parsed_set: set[str], expected: list[Any] | Any) -> float:
    """Set-aware F1 reward; exact match = 1.0."""
    if not parsed_set:
        return REWARD_MIN
    exp = {str(x).upper() for x in (expected if isinstance(expected, list) else [expected])}
    pred = {str(x).upper() for x in parsed_set}
    if not exp:
        return REWARD_MIN
    if pred == exp:
        return REWARD_MAX
    tp = len(pred & exp)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(exp)
    if precision + recall == 0:
        return REWARD_MIN
    f1 = 2.0 * precision * recall / (precision + recall)
    return max(REWARD_MIN, min(REWARD_MAX, f1))


def categorical_graded_reward(parsed: str | None, expected: Any) -> float:
    if parsed is None:
        return REWARD_MIN
    expected_norm = normalize_token(str(expected))
    return REWARD_MAX if parsed == expected_norm else REWARD_MIN


def graded_verifiable_reward(
    example: dict[str, Any],
    response: str,
) -> tuple[float, str]:
    """Compute a bounded deterministic reward and a short reason tag."""
    vtype = example["verification_type"]
    expected = example["ground_truth"]
    tolerance = example.get("tolerance")
    task_family = str(example.get("task_family", ""))

    if is_empty_or_refusal(response):
        return REWARD_MIN, "empty_or_refusal"

    if vtype == "categorical":
        parsed = parse_categorical(response)
        return categorical_graded_reward(parsed, expected), "categorical"

    if vtype == "numeric":
        parsed = parse_numeric(response)
        shaping = "default"
        if "waveform" in task_family and "numeric" in task_family:
            shaping = "waveform_rms"
        elif task_family == "spectrogram_peak_frequency":
            shaping = "spectrogram_peak"
        reward = numeric_graded_reward(parsed, float(expected), tolerance, shaping=shaping)
        return reward, f"numeric_{shaping}"

    if vtype == "set":
        channel_values = _extract_channel_dict(example.get("context", {}))
        known = set(channel_values.keys()) if channel_values else set()
        parsed_set = parse_set_membership(response, known_channels=known)
        return set_graded_reward(parsed_set, expected), "set_f1"

    if vtype == "ranking":
        channel_values = _extract_channel_dict(example.get("context", {}))
        known = set(channel_values.keys()) if channel_values else set()
        parsed = parse_ranking(response, known_channels=known)
        return ranking_graded_reward(parsed, expected if isinstance(expected, list) else [expected]), "ranking"

    return REWARD_MIN, f"unsupported:{vtype}"


def verifiable_reward(
    prompts: list[Any],
    completions: list[Any],
    verification_type: list[str],
    ground_truth: list[Any],
    context: list[Any],
    tolerance: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Binary pass/fail reward using deterministic verifiers."""
    del prompts, kwargs
    rewards: list[float] = []
    tolerances = tolerance if tolerance is not None else [None] * len(completions)

    for completion, vtype, expected, ctx, tol in zip(
        completions, verification_type, ground_truth, context, tolerances, strict=True
    ):
        example = {
            "verification_type": vtype,
            "ground_truth": _parse_json_field(expected),
            "context": _parse_json_field(ctx),
            "tolerance": _parse_json_field(tol),
        }
        response = _completion_text(completion)
        result = verify_example(example, response)
        rewards.append(1.0 if result.passed else 0.0)
    return rewards


def multimodal_verifiable_reward(
    prompts: list[Any],
    completions: list[Any],
    verification_type: list[str],
    ground_truth: list[Any],
    context: list[Any],
    tolerance: list[Any] | None = None,
    task_family: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Graded deterministic reward for multimodal RLVR in [0, 1]."""
    del prompts, kwargs
    rewards: list[float] = []
    tolerances = tolerance if tolerance is not None else [None] * len(completions)
    families = task_family if task_family is not None else [""] * len(completions)

    for completion, vtype, expected, ctx, tol, family in zip(
        completions, verification_type, ground_truth, context, tolerances, families, strict=True
    ):
        example = {
            "verification_type": vtype,
            "ground_truth": _parse_json_field(expected),
            "context": _parse_json_field(ctx),
            "tolerance": _parse_json_field(tol),
            "task_family": family,
        }
        response = _completion_text(completion)
        reward, _ = graded_verifiable_reward(example, response)
        reward = max(REWARD_MIN, min(REWARD_MAX, reward))
        if not (math.isfinite(reward)):
            reward = REWARD_MIN
        rewards.append(reward)
        MULTIMODAL_REWARD_TRACKER.record(family or vtype, reward)
    return rewards
