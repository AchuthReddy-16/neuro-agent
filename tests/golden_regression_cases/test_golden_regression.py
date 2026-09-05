"""Golden product regressions — evaluation-only; never use as training data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from neuro_agent.agent.task_plan import ArtifactContext, plan_task
from neuro_agent.api.service import UNRELIABLE_VISION_FIGURE_NOTE

TESTS_DIR = Path(__file__).resolve().parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_api_backend import MockVisionRunner, _wire_client  # noqa: E402

GOLDEN_ROOT = Path(__file__).resolve().parent
CASES_DIR = GOLDEN_ROOT / "cases"
ASSETS_DIR = GOLDEN_ROOT / "assets"

STALE_MARKERS = (
    "left_fist",
    "right_fist",
    "sample=None",
    "channels meeting the criterion",
    "highest-ranked channels",
    "Uncertainty: None",
    "Tools used:",
)


def _load_case(name: str) -> dict:
    return json.loads((CASES_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case_file",
    ["v2_topomap.json", "v3_spectrogram.json"],
)
def test_golden_case_is_evaluation_only(case_file: str):
    case = _load_case(case_file)
    assert case.get("evaluation_only") is True
    assert case.get("training_allowed") is False
    assert case["baseline_output"]["scientifically_correct"] is False
    asset = GOLDEN_ROOT / case["asset"]
    assert asset.is_file(), f"missing golden asset {asset}"


def test_explain_this_image_routes_to_vision_not_concept():
    arts = ArtifactContext(has_image=True, image_id="asset_golden", has_sample=False)
    for q in [
        "explain this image",
        "can u explain this image once what does it tells",
    ]:
        p = plan_task(q, artifacts=arts)
        assert p.use_vision, f"expected VISION for {q!r}, got {p.reason}"
        assert not p.text_only
        assert not p.use_tools


def test_explain_motor_imagery_stays_text_with_image_selected():
    arts = ArtifactContext(has_image=True, image_id="asset_golden", has_sample=False)
    p = plan_task("Explain motor imagery.", artifacts=arts)
    assert p.text_only and not p.use_vision
    assert p.reason == "concept_explanation"


@pytest.mark.parametrize(
    "case_file,baseline_key",
    [
        ("v2_topomap.json", "v2"),
        ("v3_spectrogram.json", "v3"),
    ],
)
def test_golden_v2_v3_product_contract(tmp_path, monkeypatch, case_file, baseline_key):
    """Product contract for real V2/V3: VISION, provenance, isolation, Limitations.

    Baseline output is recorded for future checkpoint comparison only — this test
    does not assert scientific correctness of the VLM text.
    """
    case = _load_case(case_file)
    asset_path = GOLDEN_ROOT / case["asset"]
    png = asset_path.read_bytes()
    baseline = case["baseline_output"]["text"]

    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    monkeypatch.setattr(svc, "enable_vlm", True)

    def _fake_vlm(question: str, path):  # noqa: ANN001
        assert path.exists()
        # Emit the recorded production baseline (not a claim of correctness).
        return baseline

    monkeypatch.setattr(svc, "_run_vlm_on_path", _fake_vlm)

    up = client.post(
        "/api/upload",
        data={
            "fileType": "figure",
            "filename": case["source_filename"],
        },
        files={"file": (case["source_filename"], png, "image/png")},
    )
    assert up.status_code == 200, up.text
    asset_id = up.json()["assetId"]
    exp_id = up.json()["experimentId"]

    r = client.post(
        "/api/analyze",
        json={
            "experimentId": exp_id,
            "question": case["question"],
            "imageId": asset_id,
            "conversationHistory": [
                {
                    "role": "user",
                    "content": "Which channels have highest beta for this sample?",
                },
                {
                    "role": "assistant",
                    "content": "T8 leads with left_fist/right_fist contrast.",
                    "tools_used": ["rank_channels_for_sample"],
                    "route": "TEXT",
                    "evidence_summary": "T8=278 left_fist=0.168",
                },
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "VISION"
    assert body.get("visionUsed") or body.get("vision_used")
    assert (body.get("sourceImageId") or body.get("source_image_id")) == asset_id
    src_name = body.get("sourceImageName") or body.get("source_image_name") or ""
    assert case["source_filename"] in src_name or src_name.endswith(case["source_filename"])
    assert body.get("tools_used") == []
    assert body.get("computed_evidence") == []
    answer = body.get("answer") or ""
    assert baseline in answer or answer.strip() == baseline.strip()
    for marker in STALE_MARKERS:
        assert marker.lower() not in answer.lower(), f"stale marker {marker!r}"
    uncertainty = body.get("uncertainty") or ""
    assert UNRELIABLE_VISION_FIGURE_NOTE in uncertainty
