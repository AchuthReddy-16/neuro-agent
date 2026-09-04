"""Typed schemas, band definitions, and errors for neuroscience tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from neuro_agent.paths import PROJECT_ROOT

SAMPLE_ID_RE = re.compile(r"^(S\d{3})_(R\d{2})_(E\d{3})$")
DEFAULT_SAMPLING_RATE_HZ = 160.0
DEFAULT_TOLERANCE = {"absolute": 1e-6, "relative": 1e-6}

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLES_PATH = PROCESSED_DATA_DIR / "samples.jsonl"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.parquet"
ARRAYS_DIR = PROCESSED_DATA_DIR / "arrays"
CONDITION_SUMMARIES_PATH = PROCESSED_DATA_DIR / "condition_summaries.parquet"
PREPROCESSING_REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "preprocessing_report.json"

# Welch PSD parameters (README_VISION_DATA.md, vision PSD pipeline).
WELCH_NPERSEG = 320
WELCH_NOVERLAP = 160
PSD_SEARCH_FMIN_HZ = 1.0
PSD_SEARCH_FMAX_HZ = 40.0
PSD_GROUP_CHANNELS = ("C3", "CZ", "C4")

# Band-limited power frequency ranges (Hz, inclusive). Column names in features.parquet.
BAND_DEFINITIONS: dict[str, dict[str, Any]] = {
    "delta": {
        "freq_hz": (1.0, 4.0),
        "feature_column": "delta_power",
        "display_name": "delta",
    },
    "theta": {
        "freq_hz": (4.0, 8.0),
        "feature_column": "theta_power",
        "display_name": "theta",
    },
    "alpha_mu": {
        "freq_hz": (8.0, 13.0),
        "feature_column": "alpha_mu_power",
        "display_name": "alpha/mu",
    },
    "beta": {
        "freq_hz": (13.0, 30.0),
        "feature_column": "beta_power",
        "display_name": "beta",
    },
}

BAND_ALIASES = {
    "delta": "delta",
    "theta": "theta",
    "alpha": "alpha_mu",
    "alpha_mu": "alpha_mu",
    "mu": "alpha_mu",
    "beta": "beta",
}

METRIC_COLUMNS = {
    "rms": "rms",
    "delta_power": "delta_power",
    "theta_power": "theta_power",
    "alpha_mu_power": "alpha_mu_power",
    "beta_power": "beta_power",
    "mean": "mean",
    "variance": "variance",
    "std": "std",
    "peak_to_peak": "peak_to_peak",
}

PowerSource = Literal["features", "recompute"]
Comparator = Literal["gt", "ge", "lt", "le"]
ThresholdMode = Literal["absolute", "median", "upper_quartile"]
RankOrder = Literal["ascending", "descending"]


class ToolError(Exception):
    """Base error for deterministic neuroscience tools."""


class SampleNotFoundError(ToolError):
    pass


class ChannelNotFoundError(ToolError):
    pass


class BandNotFoundError(ToolError):
    pass


class FeatureMissingError(ToolError):
    pass


class InvalidShapeError(ToolError):
    pass


class InvalidFrequencyRangeError(ToolError):
    pass


class FlatSpectrumError(ToolError):
    pass


class EmptyValueDictError(ToolError):
    pass


class InvalidThresholdError(ToolError):
    pass


class ConditionNotFoundError(ToolError):
    pass


class InsufficientSamplesError(ToolError):
    pass


class UnknownMetricError(ToolError):
    pass


@dataclass(frozen=True)
class BandPowerResult:
    channel: str
    band: str
    freq_hz: tuple[float, float]
    power: float
    units: str
    method: str


@dataclass(frozen=True)
class BandPowerOutput:
    sample_id: str
    results: list[BandPowerResult]
    source: PowerSource


@dataclass(frozen=True)
class RmsResult:
    channel: str
    rms: float
    units: str


@dataclass(frozen=True)
class RmsOutput:
    sample_id: str
    results: list[RmsResult]
    highest_rms_channel: str | None = None
    method: str = "epoch_rms"


@dataclass(frozen=True)
class PsdPeakResult:
    peak_frequency_hz: float
    peak_psd: float
    channel: str
    search_range_hz: tuple[float, float]
    method: str
    config: dict[str, Any] = field(default_factory=dict)
    psd_peaks_per_channel: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RankChannelsOutput:
    ranking: list[str]
    values: dict[str, float]
    order: RankOrder
    top_k: int | None
    units: str | None = None
    tie_break: str = "ascending_channel_name"


@dataclass(frozen=True)
class ThresholdSelectionOutput:
    channels: list[str]
    threshold_used: float
    comparator: Comparator
    threshold_mode: ThresholdMode
    policy: str
    n_selected: int
    units: str | None = None


@dataclass(frozen=True)
class CompareConditionsOutput:
    subject_id: str
    condition_a: str
    condition_b: str
    metric: str
    mean_a: float
    mean_b: float
    higher_condition: str
    signed_difference: float
    absolute_difference: float
    channel_values_a: dict[str, float]
    channel_values_b: dict[str, float]
    largest_absolute_difference_channel: str
    units: str
    aggregation: str


def load_channel_names() -> list[str]:
    with PREPROCESSING_REPORT_PATH.open() as handle:
        report = json.load(handle)
    channels = report.get("channel_names")
    if not channels:
        raise ToolError(f"channel_names missing in {PREPROCESSING_REPORT_PATH}")
    return [str(ch).upper() for ch in channels]


def normalize_band_name(band: str | tuple[float, float]) -> str | tuple[float, float]:
    if isinstance(band, tuple):
        return band
    key = band.strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    if key in BAND_ALIASES:
        return BAND_ALIASES[key]
    raise BandNotFoundError(f"Unknown band: {band!r}. Expected one of {sorted(BAND_DEFINITIONS)}")


def resolve_metric_column(metric: str) -> str:
    key = metric.strip().lower()
    if key in METRIC_COLUMNS:
        return METRIC_COLUMNS[key]
    if key in BAND_DEFINITIONS:
        return BAND_DEFINITIONS[key]["feature_column"]
    raise UnknownMetricError(f"Unknown metric: {metric!r}")


def power_units() -> str:
    return "uV2"


def amplitude_units() -> str:
    return "uV"


def frequency_units() -> str:
    return "Hz"
