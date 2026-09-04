"""Router, metadata, and vision evidence tests (Stage G.2)."""

from __future__ import annotations

import json
import time

import pytest

from neuro_agent.tools.evidence import ResearchToolRequest
from neuro_agent.tools.metadata import lookup_sample_metadata
from neuro_agent.tools.router import route_research_request
from neuro_agent.tools.schemas import SampleNotFoundError
from neuro_agent.tools.vision_evidence import resolve_vision_evidence


SAMPLE_ID = "S001_R01_E000"
PSD_SAMPLE = "S016_R08_E022"


class TestMetadataLookup:
    def test_lookup_by_sample_id(self) -> None:
        meta = lookup_sample_metadata(sample_id=SAMPLE_ID)
        assert meta.sample_id == SAMPLE_ID
        assert meta.subject_id == "S001"
        assert meta.run_id == "R01"
        assert meta.epoch_index == 0
        assert meta.task_type == "baseline"
        assert meta.movement == "rest"
        assert meta.condition == "baseline_rest"
        assert meta.task_type != meta.movement
        assert meta.sampling_rate_hz == 160.0
        assert len(meta.channels) == 64
        assert "C3" in meta.channels
        assert meta.array_ref.array_index == 0
        assert meta.feature_refs.feature_path.endswith("features.parquet")

    def test_lookup_by_subject_run_epoch(self) -> None:
        meta = lookup_sample_metadata(subject_id="S001", run_id="R01", epoch=0)
        assert meta.sample_id == SAMPLE_ID

    def test_missing_sample_raises(self) -> None:
        with pytest.raises(SampleNotFoundError):
            lookup_sample_metadata(sample_id="S999_R99_E999")

    def test_vision_assets_attached(self) -> None:
        meta = lookup_sample_metadata(sample_id=PSD_SAMPLE)
        assert len(meta.vision_assets) >= 1
        families = {asset.visualization_type for asset in meta.vision_assets}
        assert "power_spectral_density" in families


class TestVisionEvidence:
    def test_resolve_by_sample_psd(self) -> None:
        refs = resolve_vision_evidence(sample_id=PSD_SAMPLE, visual_type="psd")
        assert len(refs) == 1
        ref = refs[0]
        assert ref.family == "power_spectral_density"
        assert ref.source_sample_id == PSD_SAMPLE
        assert "peak_frequency_hz" in ref.source_numeric_values

    def test_resolve_by_image_id(self) -> None:
        image_id = None
        for line in open("data/processed/vision/metadata/images.jsonl"):
            record = json.loads(line)
            if record.get("visualization_type") == "waveform":
                image_id = record["image_id"]
                break
        assert image_id is not None
        refs = resolve_vision_evidence(image_id=image_id)
        assert len(refs) == 1
        assert refs[0].family == "waveform"
        assert "rms_uV" in refs[0].source_numeric_values

    def test_resolve_all_types_for_sample(self) -> None:
        refs = resolve_vision_evidence(sample_id=PSD_SAMPLE)
        families = {ref.family for ref in refs}
        assert "power_spectral_density" in families
        assert "spectrogram" in families
        assert "topomap_multi_band" in families
        assert "channel_band_power" in families
        assert "waveform" in families

    def test_missing_visual_type_raises(self) -> None:
        with pytest.raises(SampleNotFoundError):
            resolve_vision_evidence(sample_id=SAMPLE_ID, visual_type="nonexistent_viz_type")


class TestRouter:
    def test_band_power_beta_c3(self) -> None:
        request = ResearchToolRequest(
            question_type="band_power",
            sample_id=SAMPLE_ID,
            frequency_band="beta",
            channels=["C3"],
        )
        bundle = route_research_request(request)
        assert bundle.success
        assert bundle.units == "uV2"
        assert bundle.numeric_evidence["channel"] == "C3"
        assert bundle.numeric_evidence["value"] > 0
        assert any(p.source == "stored_feature" for p in bundle.provenance)
        assert any(inv.name == "compute_band_power" for inv in bundle.tool_invocations)

    def test_rms_c3(self) -> None:
        request = ResearchToolRequest(
            question_type="rms",
            sample_id=SAMPLE_ID,
            channels=["C3"],
        )
        bundle = route_research_request(request)
        assert bundle.success
        assert bundle.units == "uV"
        assert bundle.numeric_evidence["channel"] == "C3"
        assert any(p.source == "raw_eeg" for p in bundle.provenance)

    def test_psd_peak(self) -> None:
        request = ResearchToolRequest(
            question_type="psd_peak",
            sample_id=PSD_SAMPLE,
        )
        bundle = route_research_request(request)
        assert bundle.success
        assert bundle.units == "Hz"
        assert "peak_frequency_hz" in bundle.numeric_evidence

    def test_channel_ranking_top_k(self) -> None:
        request = ResearchToolRequest(
            question_type="channel_ranking",
            sample_id=SAMPLE_ID,
            frequency_band="beta",
            top_k=3,
            sort_direction="descending",
        )
        bundle = route_research_request(request)
        assert bundle.success
        assert bundle.ranked_evidence is not None
        assert len(bundle.ranked_evidence["ranking"]) == 3
        assert bundle.numeric_evidence["top_channel"] == bundle.ranked_evidence["ranking"][0]

    def test_threshold_set_upper_quartile(self) -> None:
        request = ResearchToolRequest(
            question_type="threshold_set",
            sample_id=SAMPLE_ID,
            frequency_band="beta",
            threshold_mode="upper_quartile",
        )
        bundle = route_research_request(request)
        assert bundle.success
        assert bundle.set_evidence is not None
        assert bundle.set_evidence["comparator"] == "ge"
        assert bundle.set_evidence["threshold_mode"] == "upper_quartile"
        assert bundle.set_evidence["n_selected"] >= 1

    def test_condition_comparison(self) -> None:
        request = ResearchToolRequest(
            question_type="condition_comparison",
            subject_id="S013",
            condition_a="left_fist",
            condition_b="right_fist",
            metric="alpha_mu_power",
        )
        bundle = route_research_request(request)
        assert bundle.success
        assert bundle.condition_evidence is not None
        assert "mean_a" in bundle.numeric_evidence
        assert any(p.source == "comparison" for p in bundle.provenance)

    def test_invalid_channel_error(self) -> None:
        request = ResearchToolRequest(
            question_type="rms",
            sample_id=SAMPLE_ID,
            channels=["NOTACHANNEL"],
        )
        bundle = route_research_request(request)
        assert not bundle.success
        assert bundle.error is not None

    def test_missing_sample_error(self) -> None:
        request = ResearchToolRequest(
            question_type="band_power",
            sample_id="S999_R99_E999",
            frequency_band="beta",
            channels=["C3"],
        )
        bundle = route_research_request(request)
        assert not bundle.success

    def test_vision_evidence_attachment(self) -> None:
        request = ResearchToolRequest(
            question_type="psd_peak",
            sample_id=PSD_SAMPLE,
            include_vision_evidence=True,
            requested_visual_type="psd",
        )
        bundle = route_research_request(request)
        assert bundle.success
        assert len(bundle.vision_evidence) == 1
        assert any(p.source == "vision_metadata" for p in bundle.provenance)

    def test_metadata_in_bundle(self) -> None:
        request = ResearchToolRequest(
            question_type="band_power",
            sample_id=SAMPLE_ID,
            frequency_band="beta",
            channels=["C3"],
        )
        bundle = route_research_request(request)
        assert bundle.metadata is not None
        assert bundle.metadata["task_type"] == "baseline"
        assert bundle.metadata["movement"] == "rest"


class TestRouterSmoke:
    """CPU smoke: 5–10 structured requests with latency checks."""

    SMOKE_REQUESTS = [
        ResearchToolRequest(
            question_type="channel_ranking",
            sample_id=SAMPLE_ID,
            frequency_band="beta",
            top_k=3,
        ),
        ResearchToolRequest(
            question_type="rms",
            sample_id=SAMPLE_ID,
            channels=["C3"],
        ),
        ResearchToolRequest(
            question_type="psd_peak",
            sample_id=PSD_SAMPLE,
        ),
        ResearchToolRequest(
            question_type="threshold_set",
            sample_id=SAMPLE_ID,
            frequency_band="beta",
            threshold_mode="upper_quartile",
        ),
        ResearchToolRequest(
            question_type="condition_comparison",
            subject_id="S013",
            condition_a="left_fist",
            condition_b="right_fist",
        ),
        ResearchToolRequest(
            question_type="band_power",
            sample_id=SAMPLE_ID,
            frequency_band="beta",
            channels=["C3", "C4", "CZ"],
        ),
        ResearchToolRequest(
            question_type="channel_ranking",
            sample_id=SAMPLE_ID,
            metric="rms",
            channels=["C3", "C4", "CZ"],
            top_k=2,
        ),
        ResearchToolRequest(
            question_type="threshold_set",
            sample_id=SAMPLE_ID,
            frequency_band="beta",
            threshold_mode="median",
            comparator="gt",
        ),
        ResearchToolRequest(
            question_type="psd_peak",
            sample_id=PSD_SAMPLE,
            include_vision_evidence=True,
            requested_visual_type="psd",
        ),
        ResearchToolRequest(
            question_type="band_power",
            subject_id="S001",
            run_id="R01",
            epoch=0,
            frequency_band="beta",
            channels=["C3"],
        ),
    ]

    def test_smoke_requests_succeed(self) -> None:
        latencies: list[tuple[str, float]] = []
        for request in self.SMOKE_REQUESTS:
            start = time.perf_counter()
            bundle = route_research_request(request)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append((request.question_type, elapsed_ms))
            assert bundle.success, f"{request.question_type} failed: {bundle.error}"
            assert bundle.request_id.startswith("req_")
            assert len(bundle.provenance) >= 1

        for qtype, elapsed_ms in latencies:
            assert elapsed_ms < 5000.0, f"{qtype} too slow: {elapsed_ms:.1f}ms"

        print("\nSmoke latencies (ms):")
        for qtype, elapsed_ms in latencies:
            print(f"  {qtype}: {elapsed_ms:.1f}")
