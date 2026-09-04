"""Strict selected-image → VLM asset provenance contracts."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from neuro_agent.api.service import AnalysisService
from test_api_backend import MockTextRunner, MockVisionRunner, _wire_client


PNG_A = b"\x89PNG\r\n\x1a\n" + b"WAVEFORM_STYLE_BYTES_AAAA" + b"\x00" * 16
PNG_B = b"\x89PNG\r\n\x1a\n" + b"SCATTER_PLOT_BYTES_BBBB" + b"\x00" * 16


class LeakyVisionRunner(MockVisionRunner):
    """Simulates agent attaching a DIFFERENT sample image than the user selected."""

    def _ask_impl(self, question: str, *, request_id: str | None = None) -> Any:
        t = super()._ask_impl(question, request_id=request_id)
        t.evidence_bundle = {
            "success": True,
            "numeric_evidence": [],
            "vision_evidence": [
                {
                    "image_id": "WRONG_SAMPLE_SCATTER_ID",
                    "family": "condition_comparison",
                    "image_path": "data/fake/scatter.png",
                }
            ],
        }
        return t


def _upload_figure(client: TestClient, *, filename: str, content: bytes, experiment_id: str | None = None):
    data = {"fileType": "figure", "filename": filename}
    if experiment_id:
        data["experiment_id"] = experiment_id
    r = client.post(
        "/api/upload",
        data=data,
        files={"file": (filename, content, "image/png")},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_uploaded_image_id_beats_leaky_sample_evidence(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, LeakyVisionRunner())
    up = _upload_figure(client, filename="Vertex_waves_EEG.png", content=PNG_A)
    asset_id = up["assetId"]
    exp_id = up["experimentId"]

    r = client.post(
        "/api/analyze",
        json={
            "experimentId": exp_id,
            "question": "What does this image show?",
            "imageId": asset_id,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "VISION"
    assert body.get("visionUsed") or body.get("vision_used")
    assert (body.get("sourceImageId") or body.get("source_image_id")) == asset_id
    assert (body.get("sourceImageName") or body.get("source_image_name")) == "Vertex_waves_EEG.png"
    assert body["visual_evidence"], "expected visual evidence for selected upload"
    assert body["visual_evidence"][0]["id"] == asset_id
    assert body["visual_evidence"][0]["id"] != "WRONG_SAMPLE_SCATTER_ID"


def test_two_image_provenance_does_not_cross_over(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    a = _upload_figure(client, filename="waveform_A.png", content=PNG_A)
    exp_id = a["experimentId"]
    id_a = a["assetId"]
    b = _upload_figure(client, filename="scatter_B.png", content=PNG_B, experiment_id=exp_id)
    id_b = b["assetId"]
    assert id_a != id_b

    r1 = client.post(
        "/api/analyze",
        json={
            "experimentId": exp_id,
            "question": "What does this image show?",
            "imageId": id_a,
        },
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert (body1.get("sourceImageId") or body1.get("source_image_id")) == id_a
    assert (body1.get("sourceImageName") or body1.get("source_image_name")) == "waveform_A.png"

    r2 = client.post(
        "/api/analyze",
        json={
            "experimentId": exp_id,
            "question": "What does this image show?",
            "imageId": id_b,
        },
    )
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert (body2.get("sourceImageId") or body2.get("source_image_id")) == id_b
    assert (body2.get("sourceImageName") or body2.get("source_image_name")) == "scatter_B.png"
    assert body2["visual_evidence"][0]["id"] == id_b
    assert body2["visual_evidence"][0]["id"] != id_a

    r3 = client.post(
        "/api/analyze",
        json={
            "experimentId": exp_id,
            "question": "What does this image show?",
            "imageId": id_a,
        },
    )
    assert r3.status_code == 200, r3.text
    body3 = r3.json()
    assert (body3.get("sourceImageId") or body3.get("source_image_id")) == id_a


def test_text_after_image_does_not_call_vision(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    up = _upload_figure(client, filename="fig.png", content=PNG_A)
    # Patch VLM to detect accidental calls
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("VLM should not run for text-only")

    monkeypatch.setattr(svc, "_run_vlm_on_path", _boom)
    monkeypatch.setattr(svc, "enable_vlm", True)

    r = client.post(
        "/api/analyze",
        json={
            "experimentId": up["experimentId"],
            "question": "What is EEG?",
            "imageId": up["assetId"],
            "conversationHistory": [
                {"role": "user", "content": "What does this image show?"},
                {"role": "assistant", "content": "A waveform.", "route": "VISION"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "TEXT"
    assert not (body.get("visionUsed") or body.get("vision_used"))
    assert body.get("sourceImageId") in (None, "")
    assert called["n"] == 0


def test_missing_image_id_fails_cleanly_when_vision_required(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    # Create empty experiment via API
    er = client.post("/api/experiment", json={})
    assert er.status_code == 200, er.text
    empty_id = er.json().get("experimentId") or er.json().get("id")
    r = client.post(
        "/api/analyze",
        json={"experimentId": empty_id, "question": "What does this image show?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route_detail"]["needs_input"] is True
    assert body["route_detail"]["need_kind"] == "image"


def test_unknown_image_id_errors(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    up = _upload_figure(client, filename="real.png", content=PNG_A)
    r = client.post(
        "/api/analyze",
        json={
            "experimentId": up["experimentId"],
            "question": "What does this image show?",
            "imageId": "asset_does_not_exist",
        },
    )
    assert r.status_code == 400
    detail = r.json()
    err = detail.get("detail") or detail
    blob = json.dumps(err if isinstance(err, dict) else detail)
    assert "missing_image" in blob


def test_resolve_exact_vision_asset_unit(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    up = _upload_figure(client, filename="UnitTest.png", content=PNG_A)
    resolved = svc.resolve_exact_vision_asset(
        experiment_id=up["experimentId"],
        image_id=up["assetId"],
    )
    assert resolved is not None
    assert resolved["visual"].id == up["assetId"]
    assert resolved["name"] == "UnitTest.png"
    assert resolved["origin"] == "uploaded"
    assert resolved["path"].exists()
    assert svc.resolve_exact_vision_asset(
        experiment_id=up["experimentId"],
        image_id="asset_missing",
    ) is None
