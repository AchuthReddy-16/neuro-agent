"""Deterministic EEG signal tools: band power, RMS, and PSD peak."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from neuro_agent.tools._stores import (
    FeatureStore,
    SampleStore,
    default_feature_store,
    default_sample_store,
    official_channels,
)
from neuro_agent.tools.normalization import normalize_channels, psd_display_channel
from neuro_agent.tools.schemas import (
    PSD_GROUP_CHANNELS,
    PSD_SEARCH_FMAX_HZ,
    PSD_SEARCH_FMIN_HZ,
    WELCH_NOVERLAP,
    WELCH_NPERSEG,
    BandNotFoundError,
    BandPowerOutput,
    BandPowerResult,
    ChannelNotFoundError,
    FlatSpectrumError,
    InvalidFrequencyRangeError,
    InvalidShapeError,
    PsdPeakResult,
    RmsOutput,
    RmsResult,
    amplitude_units,
    frequency_units,
    normalize_band_name,
    power_units,
)

# Reused logic sources:
# - RMS formula sqrt(mean(x^2)): validated against vision metadata rms_uV and features.parquet
# - Welch PSD 320/160 on C3/CZ/C4 group mean: README_VISION_DATA.md PSD pipeline
# - Band power feature lookup: features.parquet per-channel columns
# - Band power recompute: scipy.signal.welch integration (beta band exact; others approximate)


def _validate_channels(requested: list[str], available: list[str]) -> list[str]:
    available_set = {ch.upper() for ch in available}
    missing = [ch for ch in requested if ch.upper() not in available_set]
    if missing:
        raise ChannelNotFoundError(f"Unknown channels: {missing}")
    return [ch.upper() for ch in requested]


def _band_limits(band: str | tuple[float, float]) -> tuple[str, tuple[float, float], str]:
    if isinstance(band, tuple):
        if len(band) != 2 or band[1] <= band[0]:
            raise BandNotFoundError(f"Invalid custom frequency range: {band}")
        return "custom", (float(band[0]), float(band[1])), "custom_band"

    band_name = str(normalize_band_name(band))
    from neuro_agent.tools.schemas import BAND_DEFINITIONS

    definition = BAND_DEFINITIONS[band_name]
    return band_name, definition["freq_hz"], definition["feature_column"]


def compute_epoch_rms(epoch: np.ndarray) -> np.ndarray:
    """Root-mean-square amplitude per channel."""
    if epoch.ndim != 2:
        raise InvalidShapeError(f"Expected epoch with shape (n_channels, n_samples); got {epoch.shape}")
    return np.sqrt(np.mean(np.square(epoch, dtype=np.float64), axis=1))


def compute_welch_band_power(
    waveform: np.ndarray,
    sampling_rate_hz: float,
    freq_range_hz: tuple[float, float],
    *,
    nperseg: int = WELCH_NPERSEG,
    noverlap: int = WELCH_NOVERLAP,
) -> float:
    frequencies, psd = signal.welch(
        waveform,
        fs=sampling_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    fmin, fmax = freq_range_hz
    mask = (frequencies >= fmin) & (frequencies <= fmax)
    if not np.any(mask):
        raise InvalidFrequencyRangeError(f"No PSD bins in range {freq_range_hz}")
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def compute_band_power(
    sample_id: str,
    band: str | tuple[float, float],
    channels: list[str] | str = "all",
    *,
    source: str = "features",
    sample_store: SampleStore | None = None,
    feature_store: FeatureStore | None = None,
) -> BandPowerOutput:
    """Absolute band-limited power for one or more channels in an epoch."""
    sample_store = sample_store or default_sample_store()
    feature_store = feature_store or default_feature_store()
    all_channels = list(official_channels())
    requested = normalize_channels(channels, all_channels=all_channels)
    band_name, freq_range_hz, feature_column = _band_limits(band)

    if source not in {"features", "recompute"}:
        raise ValueError("source must be 'features' or 'recompute'")

    results: list[BandPowerResult] = []
    if source == "features":
        rows = feature_store.get_sample_features(sample_id)
        available = set(rows["channel"].str.upper())
        _validate_channels(requested, list(available))
        for channel in requested:
            value = feature_store.get_channel_value(sample_id, channel, feature_column)
            results.append(
                BandPowerResult(
                    channel=channel,
                    band=band_name,
                    freq_hz=freq_range_hz,
                    power=value,
                    units=power_units(),
                    method="features_parquet_lookup",
                )
            )
        return BandPowerOutput(sample_id=sample_id, results=results, source="features")

    epoch, epoch_channels, sampling_rate_hz = sample_store.load_epoch(sample_id)
    channel_indices = _validate_channels(requested, epoch_channels)
    index_map = {name: idx for idx, name in enumerate(epoch_channels)}
    for channel in channel_indices:
        waveform = epoch[index_map[channel]].astype(np.float64)
        power = compute_welch_band_power(waveform, sampling_rate_hz, freq_range_hz)
        results.append(
            BandPowerResult(
                channel=channel,
                band=band_name,
                freq_hz=freq_range_hz,
                power=power,
                units=power_units(),
                method="welch_trapezoid",
            )
        )
    return BandPowerOutput(sample_id=sample_id, results=results, source="recompute")


def compute_rms(
    sample_id: str,
    channels: list[str] | str = "all",
    *,
    include_highest: bool = True,
    sample_store: SampleStore | None = None,
    feature_store: FeatureStore | None = None,
    source: str = "epoch",
) -> RmsOutput:
    """RMS amplitude per channel for an epoch."""
    sample_store = sample_store or default_sample_store()
    all_channels = list(official_channels())
    requested = normalize_channels(channels, all_channels=all_channels)

    if source == "features":
        feature_store = feature_store or default_feature_store()
        results = [
            RmsResult(
                channel=channel,
                rms=feature_store.get_channel_value(sample_id, channel, "rms"),
                units=amplitude_units(),
            )
            for channel in requested
        ]
        method = "features_parquet_lookup"
    elif source == "epoch":
        epoch, epoch_channels, _ = sample_store.load_epoch(sample_id)
        channel_indices = _validate_channels(requested, epoch_channels)
        index_map = {name: idx for idx, name in enumerate(epoch_channels)}
        rms_values = compute_epoch_rms(epoch)
        results = [
            RmsResult(
                channel=channel,
                rms=float(rms_values[index_map[channel]]),
                units=amplitude_units(),
            )
            for channel in channel_indices
        ]
        method = "epoch_rms"
    else:
        raise ValueError("source must be 'epoch' or 'features'")

    highest: str | None = None
    if include_highest and results:
        highest = max(results, key=lambda item: (item.rms, item.channel)).channel
    return RmsOutput(
        sample_id=sample_id,
        results=results,
        highest_rms_channel=highest,
        method=method,
    )


def find_psd_peak(
    sample_id: str,
    *,
    channel: str | None = None,
    freq_range_hz: tuple[float, float] | None = None,
    sampling_rate_hz: float | None = None,
    sample_store: SampleStore | None = None,
    nperseg: int = WELCH_NPERSEG,
    noverlap: int = WELCH_NOVERLAP,
    use_group_mean: bool = True,
) -> PsdPeakResult:
    """Argmax frequency of Welch PSD in the requested band."""
    sample_store = sample_store or default_sample_store()
    record = sample_store.get(sample_id)
    epoch, epoch_channels, epoch_fs = sample_store.load_epoch(sample_id)
    fs = float(sampling_rate_hz or epoch_fs)
    fmin, fmax = freq_range_hz or (PSD_SEARCH_FMIN_HZ, PSD_SEARCH_FMAX_HZ)
    if fmax <= fmin:
        raise InvalidFrequencyRangeError(f"Invalid search range: {(fmin, fmax)}")

    if use_group_mean:
        channels = list(PSD_GROUP_CHANNELS)
        psd_curves: list[np.ndarray] = []
        frequencies: np.ndarray | None = None
        for ch in channels:
            ch = _validate_channels([ch], epoch_channels)[0]
            idx = epoch_channels.index(ch)
            frequencies, psd = signal.welch(
                epoch[idx],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            psd_curves.append(psd)
        group_psd = np.mean(np.stack(psd_curves, axis=0), axis=0)
        display_channel = channel or psd_display_channel(str(record["movement"]))
    else:
        if channel is None:
            raise ValueError("channel is required when use_group_mean=False")
        display_channel = _validate_channels([channel], epoch_channels)[0]
        idx = epoch_channels.index(display_channel)
        frequencies, group_psd = signal.welch(
            epoch[idx],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )

    assert frequencies is not None
    mask = (frequencies >= fmin) & (frequencies <= fmax)
    if not np.any(mask):
        raise InvalidFrequencyRangeError(f"No PSD bins in range {(fmin, fmax)}")
    band_freqs = frequencies[mask]
    band_psd = group_psd[mask]
    if band_psd.size == 0 or not np.isfinite(band_psd).any():
        raise FlatSpectrumError("PSD is empty or non-finite in search range")
    peak_idx = int(np.argmax(band_psd))
    peak_freq = float(band_freqs[peak_idx])
    peak_psd = float(band_psd[peak_idx])

    per_channel_peaks: dict[str, float] = {}
    for ch in PSD_GROUP_CHANNELS:
        if ch not in epoch_channels:
            continue
        idx = epoch_channels.index(ch)
        ch_freqs, ch_psd = signal.welch(epoch[idx], fs=fs, nperseg=nperseg, noverlap=noverlap)
        ch_mask = (ch_freqs >= fmin) & (ch_freqs <= fmax)
        per_channel_peaks[ch] = float(ch_freqs[ch_mask][int(np.argmax(ch_psd[ch_mask]))])

    return PsdPeakResult(
        peak_frequency_hz=peak_freq,
        peak_psd=peak_psd,
        channel=display_channel,
        search_range_hz=(fmin, fmax),
        method="welch_group_mean" if use_group_mean else "welch_single_channel",
        config={
            "nperseg": nperseg,
            "noverlap": noverlap,
            "sampling_rate_hz": fs,
            "group_channels": list(PSD_GROUP_CHANNELS) if use_group_mean else [display_channel],
        },
        psd_peaks_per_channel=per_channel_peaks,
    )
