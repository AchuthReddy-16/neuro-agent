"""Channel ranking and threshold selection tools."""

from __future__ import annotations

from typing import Any

import numpy as np

from neuro_agent.tools.eeg_signal import compute_band_power, compute_rms
from neuro_agent.tools.schemas import (
    Comparator,
    EmptyValueDictError,
    InvalidThresholdError,
    RankChannelsOutput,
    RankOrder,
    ThresholdMode,
    ThresholdSelectionOutput,
    UnknownMetricError,
    amplitude_units,
    power_units,
    resolve_metric_column,
)

# Reused logic sources:
# - Ranking tie-break (-value, channel asc): vision metadata channel_rankings_descending
# - Threshold upper quartile: numpy.percentile(..., 75, method='higher') + ge comparator
# - Threshold median strict-above: numpy.median + gt comparator (RLVR/SFT set tasks)
# - Verifier set semantics: verifiers.py exact set equality on channel labels


THRESHOLD_POLICIES: dict[str, dict[str, Any]] = {
    "absolute": {
        "description": "Use caller-supplied threshold value.",
        "comparator_default": "ge",
    },
    "median": {
        "description": "Threshold = numpy.median(values); RLVR/SFT uses strict gt.",
        "comparator_default": "gt",
        "compute": lambda values: float(np.median(values)),
    },
    "upper_quartile": {
        "description": "Threshold = numpy.percentile(values, 75, method='higher'); multimodal ge.",
        "comparator_default": "ge",
        "compute": lambda values: float(np.percentile(values, 75, method="higher")),
    },
}

_COMPARATORS: dict[Comparator, Any] = {
    "gt": np.greater,
    "ge": np.greater_equal,
    "lt": np.less,
    "le": np.less_equal,
}


def _metric_units(metric: str) -> str | None:
    column = resolve_metric_column(metric)
    if column == "rms":
        return amplitude_units()
    if column.endswith("_power"):
        return power_units()
    return None


def _values_from_mapping(values: dict[str, float]) -> tuple[list[str], np.ndarray]:
    if not values:
        raise EmptyValueDictError("values dict is empty")
    channels = sorted(values.keys(), key=lambda ch: ch.upper())
    array = np.array([float(values[ch]) for ch in channels], dtype=np.float64)
    if not np.isfinite(array).all():
        raise InvalidThresholdError("values contain non-finite numbers")
    return channels, array


def rank_channels(
    values: dict[str, float],
    *,
    order: RankOrder = "descending",
    top_k: int | None = None,
    units: str | None = None,
) -> RankChannelsOutput:
    """Rank channels by numeric value with deterministic tie-breaking."""
    channels, array = _values_from_mapping(values)
    channel_values = {ch: float(val) for ch, val in zip(channels, array)}

    if order == "descending":
        sorted_items = sorted(channel_values.items(), key=lambda item: (-item[1], item[0]))
    elif order == "ascending":
        sorted_items = sorted(channel_values.items(), key=lambda item: (item[1], item[0]))
    else:
        raise ValueError("order must be 'ascending' or 'descending'")

    ranking = [ch for ch, _ in sorted_items]
    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        ranking = ranking[:top_k]

    return RankChannelsOutput(
        ranking=ranking,
        values=channel_values,
        order=order,
        top_k=top_k,
        units=units,
    )


def rank_channels_for_sample(
    sample_id: str,
    metric: str,
    *,
    channels: list[str] | str = "all",
    order: RankOrder = "descending",
    top_k: int | None = None,
) -> RankChannelsOutput:
    """Convenience wrapper: compute metric for a sample then rank."""
    column = resolve_metric_column(metric)
    if column == "rms":
        output = compute_rms(sample_id, channels=channels, include_highest=False)
        values = {item.channel: item.rms for item in output.results}
        units = amplitude_units()
    elif column.endswith("_power"):
        band = column.replace("_power", "")
        output = compute_band_power(sample_id, band, channels=channels, source="features")
        values = {item.channel: item.power for item in output.results}
        units = power_units()
    else:
        raise UnknownMetricError(f"Unsupported sample metric for ranking: {metric!r}")
    return rank_channels(values, order=order, top_k=top_k, units=units)


def _resolve_threshold(
    values: dict[str, float],
    *,
    threshold: float | None,
    threshold_mode: ThresholdMode,
    comparator: Comparator | None,
) -> tuple[float, Comparator, str]:
    policy = THRESHOLD_POLICIES[threshold_mode]
    if threshold_mode == "absolute":
        if threshold is None:
            raise InvalidThresholdError("threshold is required for absolute mode")
        threshold_used = float(threshold)
        comp: Comparator = comparator or policy["comparator_default"]
        policy_name = "absolute_supplied_threshold"
    else:
        _, array = _values_from_mapping(values)
        threshold_used = policy["compute"](array)
        comp = comparator or policy["comparator_default"]
        policy_name = f"{threshold_mode}_{comp}"
    if comp not in _COMPARATORS:
        raise InvalidThresholdError(f"Unsupported comparator: {comp!r}")
    return threshold_used, comp, policy_name


def select_channels_above_threshold(
    values: dict[str, float],
    *,
    threshold: float | None = None,
    comparator: Comparator | None = None,
    threshold_mode: ThresholdMode = "absolute",
    units: str | None = None,
) -> ThresholdSelectionOutput:
    """Select channels where value {comparator} threshold."""
    channels, array = _values_from_mapping(values)
    threshold_used, comp, policy_name = _resolve_threshold(
        values,
        threshold=threshold,
        threshold_mode=threshold_mode,
        comparator=comparator,
    )
    compare = _COMPARATORS[comp]
    selected = [ch for ch, val in zip(channels, array) if bool(compare(val, threshold_used))]
    selected.sort(key=lambda ch: ch.upper())

    return ThresholdSelectionOutput(
        channels=selected,
        threshold_used=threshold_used,
        comparator=comp,
        threshold_mode=threshold_mode,
        policy=policy_name,
        n_selected=len(selected),
        units=units,
    )


def select_channels_for_multimodal_source_values(
    source_values: dict[str, Any],
) -> ThresholdSelectionOutput:
    """Apply multimodal vision task threshold semantics (at_or_above_threshold)."""
    operation = source_values.get("operation")
    if operation != "at_or_above_threshold":
        raise InvalidThresholdError(f"Unsupported multimodal operation: {operation!r}")
    values = source_values.get("values") or {}
    threshold = source_values.get("threshold")
    if threshold is None:
        return select_channels_above_threshold(
            values,
            threshold_mode="upper_quartile",
            comparator="ge",
            units=source_values.get("units"),
        )
    return select_channels_above_threshold(
        values,
        threshold=float(threshold),
        threshold_mode="absolute",
        comparator="ge",
        units=source_values.get("units"),
    )


def select_channels_for_rlvr_context(context: dict[str, Any]) -> ThresholdSelectionOutput:
    """Apply RLVR/SFT set-task semantics: strictly above supplied median threshold."""
    threshold = context.get("threshold")
    values = context.get("beta_power_uV2") or context.get("values") or {}
    if threshold is None:
        return select_channels_above_threshold(
            values,
            threshold_mode="median",
            comparator="gt",
            units=power_units(),
        )
    return select_channels_above_threshold(
        values,
        threshold=float(threshold),
        threshold_mode="absolute",
        comparator="gt",
        units=power_units(),
    )
