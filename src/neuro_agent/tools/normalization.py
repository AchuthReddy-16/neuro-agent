"""Canonical normalization for channels, movement, task type, and conditions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from neuro_agent.tools.schemas import ChannelNotFoundError

MOVEMENT_LABELS = frozenset(
    {
        "rest",
        "left_fist",
        "right_fist",
        "both_fists",
        "both_feet",
    }
)

TASK_TYPES = frozenset({"baseline", "execution", "imagery"})

CONDITION_PATTERN = re.compile(
    r"^(?P<task_type>baseline|execution|imagery)_(?P<movement>rest|left_fist|right_fist|both_fists|both_feet)$"
)


@dataclass(frozen=True)
class SampleLabels:
    """Explicit sample metadata fields (not collapsed)."""

    sample_id: str
    subject_id: str
    run_id: str
    task_type: str
    movement: str
    condition: str
    protocol: str | None = None
    event_code: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class ConditionSpec:
    """A condition selector that may be movement-only or compound."""

    movement: str | None = None
    task_type: str | None = None
    condition: str | None = None

    def describe(self) -> str:
        if self.condition:
            return self.condition
        parts = [p for p in (self.task_type, self.movement) if p]
        return "_".join(parts)


def normalize_channel(channel: str, *, valid_channels: set[str] | None = None) -> str:
    normalized = channel.strip().upper()
    if valid_channels is not None and normalized not in valid_channels:
        raise ChannelNotFoundError(f"Unknown channel: {channel!r}")
    return normalized


def normalize_channels(
    channels: list[str] | str,
    *,
    all_channels: list[str],
) -> list[str]:
    if channels == "all":
        return list(all_channels)
    valid = set(all_channels)
    return [normalize_channel(ch, valid_channels=valid) for ch in channels]


def normalize_movement(movement: str) -> str:
    normalized = movement.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in MOVEMENT_LABELS:
        raise ValueError(f"Unknown movement: {movement!r}")
    return normalized


def normalize_task_type(task_type: str) -> str:
    normalized = task_type.strip().lower()
    if normalized not in TASK_TYPES:
        raise ValueError(f"Unknown task_type: {task_type!r}")
    return normalized


def parse_compound_condition(condition: str) -> tuple[str, str]:
    match = CONDITION_PATTERN.match(condition.strip().lower())
    if not match:
        raise ValueError(f"Invalid compound condition: {condition!r}")
    return match.group("task_type"), match.group("movement")


def make_compound_condition(task_type: str, movement: str) -> str:
    return f"{normalize_task_type(task_type)}_{normalize_movement(movement)}"


def sample_labels_from_record(record: dict[str, Any]) -> SampleLabels:
    return SampleLabels(
        sample_id=str(record["sample_id"]),
        subject_id=str(record["subject_id"]),
        run_id=str(record["run_id"]),
        task_type=str(record["task_type"]),
        movement=str(record["movement"]),
        condition=str(record["condition"]),
        protocol=record.get("protocol"),
        event_code=record.get("event_code"),
        split=record.get("split"),
    )


def resolve_condition_spec(
    condition: str,
    *,
    default_task_type: str | None = None,
) -> ConditionSpec:
    """Resolve a condition string to movement/task_type/condition fields."""
    text = condition.strip().lower()
    if CONDITION_PATTERN.match(text):
        task_type, movement = parse_compound_condition(text)
        return ConditionSpec(task_type=task_type, movement=movement, condition=text)
    if text in MOVEMENT_LABELS:
        return ConditionSpec(
            movement=text,
            task_type=default_task_type,
            condition=make_compound_condition(default_task_type, text) if default_task_type else None,
        )
    raise ValueError(f"Unrecognized condition selector: {condition!r}")


def psd_display_channel(movement: str) -> str:
    """Channel selection rule from README_VISION_DATA.md spectrogram/PSD display."""
    movement_norm = normalize_movement(movement)
    if movement_norm == "right_fist":
        return "C3"
    if movement_norm == "left_fist":
        return "C4"
    return "CZ"
