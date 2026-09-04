"""Acceptance: vision answers must not include stale tool/EEG evidence."""

from __future__ import annotations

from test_api_backend import MockTextRunner, MockVisionRunner, _wire_client


PNG = b"\x89PNG\r\n\x1a\n" + b"VISION_ASSET_BYTES" + b"\x00" * 24

STALE_MARKERS = (
    "left_fist",
    "right_fist",
    "channel=",
    "sample=None",
    "Uncertainty: None",
    "raw_model_output",
    "Answer:",
    "Evidence:",
    "Tools used:",
)


class ContaminatedVisionShell(MockVisionRunner):
    """Text shell returns stale EEG/tool prose while vision is requested."""

    def ask_text_only(self, question: str, **kwargs):  # type: ignore[no-untyped-def]
        t = self._ask_impl(question)
        t.final_answer = (
            "Left_fist vs right_fist beta power: C3=0.184, C4=0.132. "
            "Channels meeting criterion: T8, IZ."
        )
        t.tool_invocations = [{"name": "rank_channels_for_sample"}]
        t.evidence_bundle = {
            "success": True,
            "numeric_evidence": [
                {"channel": "T8", "band": "beta", "value": 278.5},
            ],
            "vision_evidence": [],
            "tool_invocations": [{"name": "rank_channels_for_sample"}],
        }
        return t

    def _ask_impl(self, question: str, *, request_id: str | None = None):
        return super()._ask_impl(question, request_id=request_id)


def test_vision_answer_ignores_contaminated_text_shell(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, ContaminatedVisionShell())
    monkeypatch.setattr(svc, "enable_vlm", True)

    def _fake_vlm(question: str, path):  # noqa: ANN001
        assert path.exists()
        return "Vertex wave complexes visible over the midline electrodes."

    monkeypatch.setattr(svc, "_run_vlm_on_path", _fake_vlm)

    up = client.post(
        "/api/upload",
        data={"fileType": "figure", "filename": "Vertex_waves_EEG.png"},
        files={"file": ("Vertex_waves_EEG.png", PNG, "image/png")},
    )
    assert up.status_code == 200, up.text
    asset_id = up.json()["assetId"]
    exp_id = up.json()["experimentId"]

    r = client.post(
        "/api/analyze",
        json={
            "experimentId": exp_id,
            "question": "What do you see?",
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
    answer = body.get("answer") or ""
    assert "Vertex wave" in answer
    for marker in STALE_MARKERS:
        assert marker.lower() not in answer.lower(), f"stale marker {marker!r} in answer"
    assert "left_fist" not in answer.lower()
    assert "right_fist" not in answer.lower()
    assert "T8" not in answer
    assert body.get("tools_used") == []
    assert body.get("computed_evidence") == []
    assert (body.get("uncertainty") or "") in {"", "None"} or "None" not in (
        body.get("uncertainty") or ""
    )
    # Prefer empty uncertainty
    assert (body.get("uncertainty") or "") == ""
    assert body["verification"]["status"] == "skipped"


def test_text_after_tools_has_no_vision_evidence(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.post(
        "/api/analyze",
        json={
            "experimentId": "exp_demo_s001",
            "question": "What is EEG?",
            "conversationHistory": [
                {"role": "user", "content": "What do you see?"},
                {
                    "role": "assistant",
                    "content": "A bright left-hemisphere focus.",
                    "route": "VISION",
                },
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "TEXT"
    assert not (body.get("visionUsed") or body.get("vision_used"))
    assert body.get("visual_evidence") in ([], None)
