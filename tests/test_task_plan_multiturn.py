"""Multi-turn conversation + component planning regression tests."""

from __future__ import annotations

from neuro_agent.agent.task_plan import (
    ArtifactContext,
    ConversationTurn,
    plan_task,
)


def test_sequence_a_rank_followup_concept_greeting():
    arts = ArtifactContext(has_sample=True, sample_id="S026_R12_E000")
    h: list[ConversationTurn] = []

    p1 = plan_task(
        "Which five EEG channels have the highest beta-band power for this sample?",
        history=h,
        artifacts=arts,
    )
    assert p1.use_tools and not p1.use_vision and not p1.needs_input
    assert "TOOLS" in p1.components

    h.extend(
        [
            ConversationTurn(role="user", content=p1.resolved_question),
            ConversationTurn(
                role="assistant",
                content="The five channels are T8, IZ, O2, P7, and O1. T8 is highest.",
                tools_used=["rank_channels_for_sample"],
                evidence_summary="T8=278.57",
            ),
        ]
    )

    p2 = plan_task("Why is T8 highest?", history=h, artifacts=arts)
    assert p2.text_only and not p2.use_tools
    assert p2.is_follow_up
    assert p2.prior_context_for_answer

    h.extend(
        [
            ConversationTurn(role="user", content="Why is T8 highest?"),
            ConversationTurn(role="assistant", content="T8 shows the strongest beta power…"),
        ]
    )

    p3 = plan_task("What is beta power generally?", history=h, artifacts=arts)
    assert p3.text_only and not p3.use_tools and not p3.use_vision

    p4 = plan_task("hey", history=h, artifacts=arts)
    assert p4.text_only and not p4.use_tools
    assert p4.reason == "conversational"


def test_sequence_b_vision_then_unrelated_concept():
    arts = ArtifactContext(has_sample=True, has_image=True, image_id="asset_1")
    h: list[ConversationTurn] = []

    p1 = plan_task("What does this figure show?", history=h, artifacts=arts)
    assert p1.use_vision and not p1.use_tools

    h.extend(
        [
            ConversationTurn(role="user", content="What does this figure show?"),
            ConversationTurn(
                role="assistant",
                content="The topomap shows left-hemisphere beta focus.",
                route="VISION",
            ),
        ]
    )

    p2 = plan_task("Explain motor imagery.", history=h, artifacts=arts)
    assert p2.text_only and not p2.use_vision and not p2.use_tools


def test_sequence_c_compare_that_with_c4():
    arts = ArtifactContext(has_sample=True, sample_id="S026_R12_E000")
    h = [
        ConversationTurn(role="user", content="What is C3 beta-band power for this sample?"),
        ConversationTurn(
            role="assistant",
            content="C3 beta-band power is 12.4 μV².",
            tools_used=["compute_band_power"],
            evidence_summary="C3=12.4",
        ),
    ]
    p = plan_task("compare that with C4", history=h, artifacts=arts)
    assert p.use_tools
    assert p.is_follow_up
    assert "C3" in p.resolved_question or "previous" in p.resolved_question.lower()


def test_sequence_d_what_do_you_know_after_analysis():
    arts = ArtifactContext(has_sample=True)
    h = [
        ConversationTurn(role="user", content="Rank beta channels for this sample"),
        ConversationTurn(
            role="assistant",
            content="T8 leads…",
            tools_used=["rank_channels_for_sample"],
        ),
    ]
    p = plan_task("what do you know?", history=h, artifacts=arts)
    assert p.text_only and not p.use_tools


def test_missing_image_and_missing_data():
    p_img = plan_task(
        "What does this topomap show?",
        artifacts=ArtifactContext(has_sample=True, has_image=False),
    )
    assert p_img.needs_input and p_img.need_kind == "image"

    p_data = plan_task(
        "Which five channels have the highest beta power for this sample?",
        artifacts=ArtifactContext(has_sample=False),
    )
    assert p_data.needs_input and p_data.need_kind == "dataset"


def test_concept_beta_power_no_tools_even_with_sample_loaded():
    p = plan_task(
        "What is beta-band power?",
        artifacts=ArtifactContext(has_sample=True, has_image=True),
    )
    assert p.text_only and not p.use_tools and not p.use_vision
