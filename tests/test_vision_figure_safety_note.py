"""Vision product-safety note for unreliable open-ended figure types."""

from __future__ import annotations

from neuro_agent.api.service import (
    UNRELIABLE_VISION_FIGURE_NOTE,
    AnalysisService,
)
from neuro_agent.api.schemas import VisualEvidenceItem


def test_unreliable_figure_detection_topomap_and_spectrogram() -> None:
    assert AnalysisService._is_unreliable_open_ended_vision_figure(
        question="Describe the dominant spatial pattern",
        vision_content_type=None,
        source_image_name="scalp_topomap_beta.png",
        raw_intent={},
        visual_items=[],
    )
    assert AnalysisService._is_unreliable_open_ended_vision_figure(
        question="What stands out in the upper region?",
        vision_content_type=None,
        source_image_name="spectrogram_tf.png",
        raw_intent={"requested_visual_type": "spectrogram"},
        visual_items=[],
    )


def test_waveform_name_wins_over_stub_topomap_intent() -> None:
    assert not AnalysisService._is_unreliable_open_ended_vision_figure(
        question="What do you see?",
        vision_content_type="image/png",
        source_image_name="Vertex_waves_EEG.png",
        raw_intent={"requested_visual_type": "topomap"},
        visual_items=[],
    )


def test_waveform_figure_does_not_get_unreliable_flag() -> None:
    assert not AnalysisService._is_unreliable_open_ended_vision_figure(
        question="What do you see?",
        vision_content_type="image/png",
        source_image_name="eeg_waveform_trace.png",
        raw_intent={},
        visual_items=[
            VisualEvidenceItem(
                id="a1",
                label="waveform",
                tab="waveform",
                observation="waves",
            )
        ],
    )


def test_note_constant_is_concise() -> None:
    assert "Experimental visual reading" in UNRELIABLE_VISION_FIGURE_NOTE
    assert "not reliable for research conclusions" in UNRELIABLE_VISION_FIGURE_NOTE
    assert len(UNRELIABLE_VISION_FIGURE_NOTE) < 160
