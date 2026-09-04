"""Typed analysis result contract tests + leakage guards."""

from __future__ import annotations

from neuro_agent.api.result_contracts import build_analysis_results_payload, empty_analysis_results


def test_empty_slots_idle():
    empty = empty_analysis_results()
    for key in ("waveform", "psd", "spectrogram", "band_power", "topomap", "comparison"):
        assert empty[key]["status"] == "idle"
        assert empty[key]["payload"] is None


def test_text_only_does_not_fill_slots():
    out = build_analysis_results_payload(
        question="hey",
        route="TEXT",
        tools_used=[],
        computed_evidence=[],
        visual_evidence=[],
        answer="Hello",
        sample_id="S1",
        experiment_id="exp_1",
        task_plan={"text_only": True, "use_tools": False, "use_vision": False},
        vlm_text=None,
        image_id=None,
    )
    assert all(out[k]["status"] == "idle" for k in ("psd", "waveform", "band_power"))


def test_band_power_slot_only():
    out = build_analysis_results_payload(
        question="rank beta",
        route="TEXT",
        tools_used=["rank_channels_for_sample"],
        computed_evidence=[{"label": "Rank 1 · T8", "value": "278"}],
        visual_evidence=[],
        answer="T8 is highest",
        sample_id="S1",
        experiment_id="exp_1",
        task_plan={"text_only": False, "use_tools": True},
        vlm_text=None,
        image_id=None,
    )
    assert out["band_power"]["status"] == "ready"
    assert out["psd"]["status"] == "idle"
    assert out["vision_interpretation"]["status"] == "idle"


def test_vision_does_not_fill_psd():
    out = build_analysis_results_payload(
        question="what does this figure show",
        route="VISION",
        tools_used=[],
        computed_evidence=[],
        visual_evidence=[
            {"id": "img1", "label": "figure", "tab": "figure", "imageUrl": "http://x/a.png"}
        ],
        answer="A topomap",
        sample_id="S1",
        experiment_id="exp_1",
        task_plan={"use_vision": True, "text_only": False},
        vlm_text="A topomap",
        image_id="img1",
    )
    assert out["vision_interpretation"]["status"] == "ready"
    assert out["psd"]["status"] == "idle"
    assert out["waveform"]["status"] == "idle"
    assert out["spectrogram"]["status"] == "idle"


def test_psd_visual_evidence_only_fills_psd():
    out = build_analysis_results_payload(
        question="show psd",
        route="TEXT",
        tools_used=[],
        computed_evidence=[],
        visual_evidence=[
            {"id": "p1", "label": "PSD", "tab": "psd", "imageUrl": "http://x/psd.png"}
        ],
        answer="PSD plot",
        sample_id="S1",
        experiment_id="exp_1",
        task_plan={"use_tools": True},
        vlm_text=None,
        image_id=None,
    )
    assert out["psd"]["status"] == "ready"
    assert out["psd"]["payload"]["imageUrl"] == "http://x/psd.png"
    assert out["topomap"]["status"] == "idle"


def test_comparison_provenance_has_both_samples():
    out = build_analysis_results_payload(
        question="compare conditions",
        route="TEXT",
        tools_used=["compare_conditions"],
        computed_evidence=[{"label": "A vs B", "value": "A higher"}],
        visual_evidence=[],
        answer="A is higher",
        sample_id="S1",
        experiment_id="exp_1",
        task_plan={"use_tools": True},
        vlm_text=None,
        image_id=None,
    )
    assert out["comparison"]["status"] == "ready"
    prov = out["comparison"]["provenance"]
    assert prov.get("sample_id_a") == "S1"
    assert prov.get("sample_id_b") == "S1"
    assert prov.get("metric") == "condition_comparison"
