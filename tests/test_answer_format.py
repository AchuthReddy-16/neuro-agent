"""Product-behavior: answer composition and presentation contract."""

from __future__ import annotations

from neuro_agent.agent.answer_format import format_grounded_answer, format_grounded_parts
from neuro_agent.api.service import AnalysisService
from neuro_agent.tools.evidence import EvidenceBundle, ToolInvocation


def _inv(name: str) -> ToolInvocation:
    return ToolInvocation(
        name=name,
        inputs={},
        outputs={},
        runtime_ms=1.0,
        success=True,
    )


def _ranking_bundle(*, top_k: int = 5) -> EvidenceBundle:
    ranking = ["T8", "IZ", "O2", "P7", "O1"]
    values = {
        "T8": 278.5670471191406,
        "IZ": 239.53125,
        "O2": 212.8015899658203,
        "P7": 208.7153778076172,
        "O1": 207.8683624267578,
    }
    return EvidenceBundle(
        request_id="req_test_rank",
        question_type="channel_ranking",
        metadata={"sample_id": "S026_R12_epoch000"},
        ranked_evidence={
            "ranking": ranking,
            "values": values,
            "metric": "beta_power",
            "top_k": top_k,
            "units": "uV2",
        },
        units="uV2",
        tool_invocations=[_inv("rank_channels_for_sample")],
    )


def test_five_channel_ranking_natural_language_synthesis():
    q = "Which five EEG channels have the highest beta-band power for this sample?"
    parts = format_grounded_parts(q, _ranking_bundle())
    answer = parts.answer
    assert "Answer:" not in answer
    assert "Evidence:" not in answer
    assert "Tools used:" not in answer
    for ch in ("T8", "IZ", "O2", "P7", "O1"):
        assert ch in answer
    assert "278.57" in answer
    assert "239.53" in answer
    assert "μV²" in answer
    assert "5" in answer or "five" in answer.lower()
    assert "followed by" in answer


def test_grounded_answer_avoids_section_dump_when_no_uncertainty():
    text = format_grounded_answer(
        "Which five channels have the highest beta-band power?",
        _ranking_bundle(),
    )
    assert not text.startswith("Answer:")
    assert "Tools used:" not in text


def test_present_filters_raw_model_output_from_uncertainty():
    svc = AnalysisService.__new__(AnalysisService)
    answer, unc = svc._present_user_facing_answer(
        "Beta power is highest at T8.\nUncertainty: None",
        [
            'raw_model_output={"requires_vision": false, "question_type": "channel_ranking"}',
            '{"requires_vision": false, "question_type": "band_power"}',
            "Band definition may vary by study.",
        ],
    )
    assert "raw_model_output" not in unc
    assert "requires_vision" not in unc
    assert "Band definition may vary" in unc
    assert "Answer:" not in answer
    assert answer.startswith("Beta power")


def test_present_strips_legacy_section_labels():
    svc = AnalysisService.__new__(AnalysisService)
    answer, unc = svc._present_user_facing_answer(
        "Answer: T8 leads.\nEvidence: T8=1.2\nTools used: rank\nUncertainty: low n",
        [],
    )
    assert not answer.lower().startswith("answer:")
    assert "Evidence:" not in answer
    assert "Tools used:" not in answer
    assert "low n" in unc


def test_condition_comparison_natural_language():
    bundle = EvidenceBundle(
        request_id="req_test_cmp",
        question_type="condition_comparison",
        metadata={"sample_id": "S026_R12_epoch000"},
        condition_evidence={
            "condition_a": "left_fist",
            "condition_b": "right_fist",
            "value_a": 10.5,
            "value_b": 8.2,
            "higher_condition": "left_fist",
        },
        units="uV2",
        tool_invocations=[_inv("compare_conditions")],
    )
    parts = format_grounded_parts("Compare left_fist vs right_fist beta power", bundle)
    assert "left_fist" in parts.answer
    assert "right_fist" in parts.answer
    assert "Answer:" not in parts.answer


def test_extract_ranked_computed_evidence_is_compact():
    svc = AnalysisService.__new__(AnalysisService)
    items = svc._extract_computed_evidence(
        {
            "ranked_evidence": {
                "ranking": ["T8", "IZ"],
                "values": {"T8": 278.57, "IZ": 239.53},
                "units": "uV2",
            }
        },
        ["rank_channels_for_sample"],
    )
    assert len(items) == 2
    assert items[0].label.startswith("Rank 1")
    assert "T8" in items[0].label
    assert items[0].value == "278.57"
