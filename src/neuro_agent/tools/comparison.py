"""Condition comparison tools."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from neuro_agent.tools._stores import FeatureStore, default_feature_store
from neuro_agent.tools.normalization import (
    MOVEMENT_LABELS,
    make_compound_condition,
    normalize_movement,
    normalize_task_type,
    resolve_condition_spec,
)
from neuro_agent.tools.schemas import (
    CONDITION_SUMMARIES_PATH,
    CompareConditionsOutput,
    ConditionNotFoundError,
    InsufficientSamplesError,
    power_units,
    resolve_metric_column,
)

# Reused logic sources:
# - Subject-level movement means: vision metadata condition_comparison channel_values_*
# - Mean aggregate across channels: mean(channel_values_a) == mean_a in images.jsonl
# - Largest absolute difference channel: argmax |a-b| per channel


def _metric_units(metric_column: str) -> str:
    if metric_column.endswith("_power"):
        return power_units()
    if metric_column == "rms":
        return "uV"
    return "unitless"


def _filter_features_by_condition(
    frame: pd.DataFrame,
    *,
    subject_id: str,
    condition_spec,
) -> pd.DataFrame:
    subset = frame[frame["subject_id"] == subject_id]
    if subset.empty:
        raise ConditionNotFoundError(f"No feature rows for subject {subject_id}")

    if condition_spec.condition and "condition" in subset.columns:
        exact = subset[subset["condition"].str.lower() == condition_spec.condition]
        if not exact.empty:
            return exact
    if condition_spec.movement:
        subset = subset[subset["movement"].str.lower() == condition_spec.movement]
    if condition_spec.task_type and "task_type" in subset.columns:
        subset = subset[subset["task_type"].str.lower() == condition_spec.task_type]
    if subset.empty:
        raise ConditionNotFoundError(
            f"No samples for subject={subject_id}, condition={condition_spec.describe()}"
        )
    return subset


def compare_conditions(
    subject_id: str,
    condition_a: str,
    condition_b: str,
    *,
    metric: str = "alpha_mu_power",
    aggregation: str = "mean_across_epochs",
    feature_store: FeatureStore | None = None,
    include_mean_aggregate: bool = True,
) -> CompareConditionsOutput:
    """Compare two conditions for a subject on a per-channel metric."""
    if aggregation != "mean_across_epochs":
        raise ValueError("Only aggregation='mean_across_epochs' is supported")

    feature_store = feature_store or default_feature_store()
    metric_column = resolve_metric_column(metric)
    frame = feature_store._load()

    spec_a = resolve_condition_spec(condition_a)
    spec_b = resolve_condition_spec(condition_b)

    rows_a = _filter_features_by_condition(frame, subject_id=subject_id, condition_spec=spec_a)
    rows_b = _filter_features_by_condition(frame, subject_id=subject_id, condition_spec=spec_b)

    if rows_a["sample_id"].nunique() < 1 or rows_b["sample_id"].nunique() < 1:
        raise InsufficientSamplesError(
            f"Insufficient epochs for subject {subject_id}: "
            f"A={rows_a['sample_id'].nunique()}, B={rows_b['sample_id'].nunique()}"
        )

    channel_values_a = {
        str(channel): float(value)
        for channel, value in rows_a.groupby("channel")[metric_column].mean().items()
    }
    channel_values_b = {
        str(channel): float(value)
        for channel, value in rows_b.groupby("channel")[metric_column].mean().items()
    }
    common_channels = sorted(set(channel_values_a) & set(channel_values_b))
    if not common_channels:
        raise InsufficientSamplesError("No overlapping channels between conditions")

    diffs = {
        ch: channel_values_a[ch] - channel_values_b[ch]
        for ch in common_channels
    }
    abs_diffs = {ch: abs(delta) for ch, delta in diffs.items()}
    largest_channel = max(abs_diffs, key=lambda ch: (abs_diffs[ch], ch))

    mean_a = float(np.mean(list(channel_values_a.values()))) if include_mean_aggregate else float("nan")
    mean_b = float(np.mean(list(channel_values_b.values()))) if include_mean_aggregate else float("nan")
    signed_difference = mean_a - mean_b
    absolute_difference = abs(signed_difference)
    higher = condition_a if mean_a >= mean_b else condition_b

    return CompareConditionsOutput(
        subject_id=subject_id,
        condition_a=normalize_condition_label(condition_a, spec_a),
        condition_b=normalize_condition_label(condition_b, spec_b),
        metric=metric_column,
        mean_a=mean_a,
        mean_b=mean_b,
        higher_condition=higher,
        signed_difference=signed_difference,
        absolute_difference=absolute_difference,
        channel_values_a=channel_values_a,
        channel_values_b=channel_values_b,
        largest_absolute_difference_channel=largest_channel,
        units=_metric_units(metric_column),
        aggregation=aggregation,
    )


def normalize_condition_label(condition: str, spec) -> str:
    text = condition.strip().lower()
    if text in MOVEMENT_LABELS:
        return text
    if spec.task_type and spec.movement:
        return make_compound_condition(spec.task_type, spec.movement)
    return text


def lookup_condition_summary(
    *,
    split: str,
    task_type: str,
    movement: str,
    condition: str,
    channel: str,
    metric: str,
    summaries_path=CONDITION_SUMMARIES_PATH,
) -> dict[str, Any]:
    """Optional lookup against precomputed condition_summaries.parquet."""
    metric_column = resolve_metric_column(metric)
    frame = pd.read_parquet(summaries_path)
    match = frame[
        (frame["split"] == split)
        & (frame["task_type"] == normalize_task_type(task_type))
        & (frame["movement"] == normalize_movement(movement))
        & (frame["condition"] == condition)
        & (frame["channel"] == channel.upper())
    ]
    if match.empty:
        raise ConditionNotFoundError("Condition summary row not found")
    row = match.iloc[0]
    return {
        "mean": float(row[metric_column]),
        "std": float(row.get(f"{metric_column}_std", np.nan)),
        "count": int(row.get("count", row.get("n", 0))),
    }
