"""Unit tests for deterministic neuroscience tools (Stage G.1)."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest
from scipy import signal

from neuro_agent.tools.comparison import compare_conditions
from neuro_agent.tools.eeg_signal import (
    compute_band_power,
    compute_epoch_rms,
    compute_rms,
    compute_welch_band_power,
    find_psd_peak,
)
from neuro_agent.tools.normalization import make_compound_condition, psd_display_channel
from neuro_agent.tools.ranking import (
    rank_channels,
    select_channels_above_threshold,
    select_channels_for_multimodal_source_values,
    select_channels_for_rlvr_context,
)
from neuro_agent.tools.schemas import (
    BAND_DEFINITIONS,
    ChannelNotFoundError,
    InvalidShapeError,
    SampleNotFoundError,
)


FS = 160.0
N_SAMPLES = 640


def _sine_epoch(
    frequency_hz: float,
    *,
    amplitude: float = 10.0,
    n_channels: int = 3,
) -> np.ndarray:
    t = np.arange(N_SAMPLES, dtype=np.float64) / FS
    wave = amplitude * np.sin(2 * math.pi * frequency_hz * t)
    return np.tile(wave, (n_channels, 1)).astype(np.float32)


class TestSyntheticSignalTools:
    def test_rms_constant_signal(self) -> None:
        epoch = np.full((2, N_SAMPLES), 5.0, dtype=np.float32)
        rms = compute_epoch_rms(epoch)
        np.testing.assert_allclose(rms, [5.0, 5.0], rtol=0, atol=1e-6)

    def test_rms_sine_matches_formula(self) -> None:
        epoch = _sine_epoch(10.0, amplitude=8.0, n_channels=1)
        expected = float(np.sqrt(np.mean(epoch[0] ** 2)))
        actual = float(compute_epoch_rms(epoch)[0])
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)

    def test_psd_peak_10hz_sinusoid(self) -> None:
        epoch = _sine_epoch(10.0, amplitude=12.0, n_channels=3)
        waveform = epoch[0].astype(np.float64)
        frequencies, psd = signal.welch(waveform, fs=FS, nperseg=320, noverlap=160)
        mask = (frequencies >= 1.0) & (frequencies <= 40.0)
        peak = float(frequencies[mask][int(np.argmax(psd[mask]))])
        assert math.isclose(peak, 10.0, abs_tol=0.5)

    def test_band_power_beta_contains_13hz_energy(self) -> None:
        epoch = _sine_epoch(15.0, amplitude=6.0, n_channels=1)
        beta_power = compute_welch_band_power(epoch[0], FS, BAND_DEFINITIONS["beta"]["freq_hz"])
        total_power = compute_welch_band_power(epoch[0], FS, (1.0, 40.0))
        assert beta_power > 0.0
        assert beta_power <= total_power

    def test_invalid_epoch_shape(self) -> None:
        with pytest.raises(InvalidShapeError):
            compute_epoch_rms(np.ones((4, 8, 2), dtype=np.float32))


class TestRankingAndThreshold:
    def test_rank_channels_descending_with_ties(self) -> None:
        values = {"C3": 10.0, "C4": 10.0, "CZ": 12.0}
        output = rank_channels(values, order="descending", top_k=3)
        assert output.ranking == ["CZ", "C3", "C4"]

    def test_rank_channels_ascending(self) -> None:
        values = {"AF3": 2.0, "AF4": 1.0, "AF7": 1.0}
        output = rank_channels(values, order="ascending")
        assert output.ranking[0] == "AF4"
        assert output.ranking[1] == "AF7"

    def test_upper_quartile_threshold_boundary(self) -> None:
        values = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        threshold = float(np.percentile(list(values.values()), 75, method="higher"))
        output = select_channels_above_threshold(
            values,
            threshold_mode="upper_quartile",
            comparator="ge",
        )
        assert output.threshold_used == threshold
        assert set(output.channels) == {ch for ch, val in values.items() if val >= threshold}

    def test_median_strict_above_policy(self) -> None:
        values = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        output = select_channels_above_threshold(
            values,
            threshold_mode="median",
            comparator="gt",
        )
        assert output.comparator == "gt"
        assert set(output.channels) == {"C", "D"}

    def test_absolute_threshold_exact_boundary(self) -> None:
        values = {"A": 5.0, "B": 10.0, "C": 15.0}
        ge = select_channels_above_threshold(values, threshold=10.0, comparator="ge")
        gt = select_channels_above_threshold(values, threshold=10.0, comparator="gt")
        assert set(ge.channels) == {"B", "C"}
        assert set(gt.channels) == {"C"}


class TestRealDataValidation:
    SAMPLE_ID = "S001_R01_E000"
    PSD_SAMPLE = "S016_R08_E022"

    @classmethod
    @pytest.fixture(scope="class")
    def features_row(cls):
        import pandas as pd

        frame = pd.read_parquet("data/processed/features.parquet")
        return frame[(frame["sample_id"] == cls.SAMPLE_ID) & (frame["channel"] == "C3")].iloc[0]

    def test_rms_matches_features(self, features_row) -> None:
        output = compute_rms(self.SAMPLE_ID, channels=["C3"], source="epoch")
        assert math.isclose(output.results[0].rms, float(features_row["rms"]), rel_tol=0, abs_tol=1e-4)

    def test_band_power_features_lookup(self, features_row) -> None:
        output = compute_band_power(self.SAMPLE_ID, "beta", channels=["C3"], source="features")
        assert math.isclose(output.results[0].power, float(features_row["beta_power"]), rel_tol=0, abs_tol=1e-6)

    def test_band_power_recompute_beta_close(self, features_row) -> None:
        output = compute_band_power(self.SAMPLE_ID, "beta", channels=["C3"], source="recompute")
        rel_err = abs(output.results[0].power - float(features_row["beta_power"])) / float(
            features_row["beta_power"]
        )
        assert rel_err < 1e-5

    def test_psd_peak_matches_vision_metadata(self) -> None:
        expected = None
        for line in open("data/processed/vision/metadata/images.jsonl"):
            record = json.loads(line)
            if record.get("visualization_type") != "power_spectral_density":
                continue
            if record.get("epoch_sample_id") == self.PSD_SAMPLE:
                expected = float(record["source_numeric_values"]["peak_frequency_hz"])
                break
        assert expected is not None
        result = find_psd_peak(self.PSD_SAMPLE)
        assert math.isclose(result.peak_frequency_hz, expected, abs_tol=0.01)

    def test_multimodal_set_tasks_match_ground_truth(self) -> None:
        matched = 0
        total = 0
        for line in open("data/processed/vision/multimodal_eval_heldout.jsonl"):
            example = json.loads(line)
            if example.get("verification_type") != "set":
                continue
            total += 1
            output = select_channels_for_multimodal_source_values(example["source_values"])
            expected = {str(ch).upper() for ch in example["ground_truth"]}
            assert set(output.channels) == expected
            matched += 1
        assert total == 58
        assert matched == 58

    def test_compare_conditions_matches_vision_metadata(self) -> None:
        for line in open("data/processed/vision/metadata/images.jsonl"):
            record = json.loads(line)
            if record.get("visualization_type") != "condition_comparison":
                continue
            snv = record["source_numeric_values"]
            output = compare_conditions(
                record["subject_id"],
                snv["condition_a"],
                snv["condition_b"],
                metric="alpha_mu_power",
            )
            assert math.isclose(output.mean_a, float(snv["mean_a"]), rel_tol=0, abs_tol=1e-5)
            assert math.isclose(output.mean_b, float(snv["mean_b"]), rel_tol=0, abs_tol=1e-5)
            assert output.largest_absolute_difference_channel == snv["largest_absolute_difference_channel"]
            break

    def test_channel_validation_error(self) -> None:
        with pytest.raises(ChannelNotFoundError):
            compute_rms(self.SAMPLE_ID, channels=["NOTACHANNEL"])

    def test_sample_not_found(self) -> None:
        with pytest.raises(SampleNotFoundError):
            compute_rms("S999_R99_E999", channels=["C3"])


class TestNormalizationHelpers:
    def test_compound_condition(self) -> None:
        assert make_compound_condition("execution", "left_fist") == "execution_left_fist"

    def test_psd_display_channel_rule(self) -> None:
        assert psd_display_channel("right_fist") == "C3"
        assert psd_display_channel("left_fist") == "C4"
        assert psd_display_channel("rest") == "CZ"


class TestLatencySmoke:
    def test_tool_latency_smoke(self) -> None:
        sample_id = "S001_R01_E000"
        timings: dict[str, float] = {}

        start = time.perf_counter()
        compute_band_power(sample_id, "beta", channels=["C3"], source="features")
        timings["compute_band_power"] = time.perf_counter() - start

        start = time.perf_counter()
        compute_rms(sample_id, channels=["C3"], source="epoch")
        timings["compute_rms"] = time.perf_counter() - start

        start = time.perf_counter()
        find_psd_peak("S016_R08_E022")
        timings["find_psd_peak"] = time.perf_counter() - start

        values = {"C3": 1.0, "C4": 2.0, "CZ": 3.0}
        start = time.perf_counter()
        rank_channels(values, top_k=2)
        timings["rank_channels"] = time.perf_counter() - start

        start = time.perf_counter()
        select_channels_above_threshold(values, threshold=1.5, comparator="gt")
        timings["select_channels_above_threshold"] = time.perf_counter() - start

        start = time.perf_counter()
        compare_conditions("S013", "left_fist", "right_fist", metric="alpha_mu_power")
        timings["compare_conditions"] = time.perf_counter() - start

        for name, elapsed in timings.items():
            assert elapsed < 5.0, f"{name} too slow: {elapsed:.3f}s"
