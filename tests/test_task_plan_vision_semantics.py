"""Semantic visual-task routing — not artifact-noun phrase matching."""

from __future__ import annotations

from neuro_agent.agent.task_plan import ArtifactContext, ConversationTurn, plan_task


def test_selected_image_visual_task_varied_wording():
    arts = ArtifactContext(has_image=True, image_id="asset_a", has_sample=True)
    visual_asks = [
        "What do you see?",
        "Describe the dominant pattern on the left side",
        "Is there asymmetry between the hemispheres?",
        "What stands out in the central region?",
        "Interpret this",
        "What does this show?",
        "Can you make sense of the brighter areas?",
    ]
    for q in visual_asks:
        p = plan_task(q, artifacts=arts)
        assert p.use_vision, f"expected VISION for {q!r}, got {p.reason}"
        assert not p.use_tools
        assert not p.needs_input


def test_selected_image_does_not_hijack_concepts():
    arts = ArtifactContext(has_image=True, image_id="asset_a", has_sample=True)
    for q in [
        "What is EEG?",
        "What is beta-band power?",
        "Explain motor imagery.",
        "hey",
        "what do you know?",
    ]:
        p = plan_task(q, artifacts=arts)
        assert p.text_only and not p.use_vision, f"expected TEXT for {q!r}, got {p.reason}"


def test_visual_task_without_image_needs_input():
    arts = ArtifactContext(has_image=False, has_sample=True)
    for q in [
        "What do you see in the central region?",
        "Describe the dominant pattern",
        "Is there asymmetry between the hemispheres?",
    ]:
        p = plan_task(q, artifacts=arts)
        assert p.needs_input and p.need_kind == "image", f"expected NEEDS_INPUT for {q!r}"


def test_vision_follow_up_then_unrelated_greeting():
    arts = ArtifactContext(has_image=True, image_id="asset_a")
    h = [
        ConversationTurn(role="user", content="What do you see?"),
        ConversationTurn(
            role="assistant",
            content="Bright focus over left sensorimotor cortex.",
            route="VISION",
        ),
    ]
    p_fu = plan_task("What about the right side?", history=h, artifacts=arts)
    assert p_fu.use_vision and not p_fu.use_tools

    h2 = h + [
        ConversationTurn(role="user", content="What about the right side?"),
        ConversationTurn(role="assistant", content="Right side is cooler.", route="VISION"),
    ]
    p_hi = plan_task("hey", history=h2, artifacts=arts)
    assert p_hi.text_only and not p_hi.use_vision


def test_eeg_tools_vs_vision_when_both_loaded():
    arts = ArtifactContext(has_sample=True, has_image=True, sample_id="S1", image_id="a1")
    p_vis = plan_task("What stands out in the upper panel?", artifacts=arts)
    assert p_vis.use_vision and not p_vis.use_tools

    p_tool = plan_task(
        "Which five channels have the highest beta-band power for this sample?",
        artifacts=arts,
    )
    assert p_tool.use_tools and not p_tool.use_vision

    p_miss = plan_task(
        "Which five channels have the highest beta-band power for this sample?",
        artifacts=ArtifactContext(has_sample=False, has_image=True),
    )
    assert p_miss.needs_input and p_miss.need_kind == "dataset"


def test_ambiguous_stays_text():
    arts = ArtifactContext(has_image=True, has_sample=True)
    p = plan_task("interesting", artifacts=arts)
    assert p.text_only and not p.use_vision and not p.use_tools
