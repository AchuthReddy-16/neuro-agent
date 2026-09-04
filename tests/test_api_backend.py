"""Backend API tests — no GPU required (mocked agent)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from neuro_agent.api.dependencies import reset_singletons
from neuro_agent.api.app import create_app
from neuro_agent.api.service import AnalysisService, RuntimeState
from neuro_agent.api.experiment_store import ExperimentStore
from neuro_agent.agent.traces import AgentTrace


@dataclass
class _FakeTrace:
    request_id: str = "req_test"
    original_question: str = ""
    parsed_intent: dict[str, Any] | None = None
    intent_valid: bool = True
    routing_result: dict[str, Any] | None = None
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    evidence_bundle: dict[str, Any] | None = None
    final_answer: str | None = "C3 has the highest beta power."
    runtime_ms: float = 100.0
    intent_latency_ms: float = 10.0
    answer_latency_ms: float = 50.0
    peak_vram_mb: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verification_triggered: bool = False
    trigger_reason: list[str] = field(default_factory=list)
    first_pass_verification: dict[str, Any] | None = None
    recovery: Any | None = None
    final_verification: dict[str, Any] | None = None
    verifier_latency_ms: float = 0.0
    recovery_latency_ms: float = 0.0


class MockTextRunner:
    def ask(self, question: str, *, request_id: str | None = None) -> Any:
        return _FakeTrace(
            request_id=request_id or "req_text",
            original_question=question,
            parsed_intent={
                "question_type": "channel_ranking",
                "requires_vision": False,
                "sample_id": "S001_R01_E000",
            },
            tool_invocations=[{"name": "rank_channels_for_sample"}],
            evidence_bundle={
                "success": True,
                "numeric_evidence": [
                    {
                        "channel": "C3",
                        "band": "beta",
                        "metric": "band_power",
                        "value": 0.184,
                        "units": "a.u.",
                    }
                ],
                "ranked_evidence": {"ranking": ["C3", "C4"], "values": {"C3": 0.184, "C4": 0.1}},
                "vision_evidence": [],
            },
            final_answer="C3 ranks highest for beta power. Uncertainty: moderate.",
            verification_triggered=False,
        )


class MockVisionRunner:
    def ask(self, question: str, *, request_id: str | None = None) -> Any:
        return _FakeTrace(
            request_id=request_id or "req_vision",
            original_question=question,
            parsed_intent={
                "question_type": "visual_inspection",
                "requires_vision": True,
                "requested_visual_type": "topomap",
                "include_vision_evidence": True,
                "sample_id": "S001_R01_E000",
            },
            tool_invocations=[{"name": "resolve_vision_evidence"}],
            evidence_bundle={
                "success": True,
                "numeric_evidence": [],
                "vision_evidence": [],
            },
            final_answer="Topomap shows contralateral beta focus.",
            verification_triggered=True,
            trigger_reason=["numeric_mismatch"],
            final_verification={"passed": False},
            verifier_latency_ms=12.0,
            recovery=_Recovery(),
            recovery_latency_ms=20.0,
        )


@dataclass
class _Recovery:
    def to_dict(self) -> dict[str, Any]:
        return {"attempted": True}


class MockVerifierRunner:
    """Text path with verifier + recovery shape."""

    def ask(self, question: str, *, request_id: str | None = None) -> Any:
        return _FakeTrace(
            request_id=request_id or "req_ver",
            original_question=question,
            parsed_intent={"requires_vision": False, "question_type": "band_power"},
            tool_invocations=[{"name": "compute_band_power"}],
            evidence_bundle={
                "success": True,
                "numeric_evidence": [
                    {"channel": "C3", "band": "beta", "metric": "band_power", "value": 1.2}
                ],
            },
            final_answer="Recovered answer with C3 beta=1.2",
            verification_triggered=True,
            trigger_reason=["ungrounded_claim"],
            final_verification={"passed": True},
            verifier_latency_ms=15.0,
            recovery=_Recovery(),
            recovery_latency_ms=30.0,
        )


@pytest.fixture()
def tmp_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_singletons()
    monkeypatch.setenv("NEURO_API_STORE_ROOT", str(tmp_path / "exps"))
    monkeypatch.setenv("NEURO_API_ENABLE_VLM", "0")
    monkeypatch.setenv("NEURO_API_LOAD_AGENT", "0")
    reset_singletons()
    yield tmp_path
    reset_singletons()


@pytest.fixture()
def client_text(tmp_store, monkeypatch: pytest.MonkeyPatch):
    from neuro_agent.api import dependencies as deps

    reset_singletons()
    store = ExperimentStore(root=tmp_store / "exps")
    svc = AnalysisService(store=store, mock_runner=MockTextRunner(), enable_vlm=False)
    monkeypatch.setattr(deps, "get_store", lambda: store)
    monkeypatch.setattr(deps, "get_service", lambda: svc)
    # bypass lru_cache wrappers
    deps.get_store.cache_clear()
    deps.get_service.cache_clear()
    app = create_app()
    # re-bind after create_app cached settings
    with TestClient(app) as c:
        # ensure service used by routes is our mock
        monkeypatch.setattr(deps, "get_service", lambda: svc)
        monkeypatch.setattr(deps, "get_store", lambda: store)
        yield c, svc, store


def _wire_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner) -> tuple[TestClient, AnalysisService, ExperimentStore]:
    from neuro_agent.api import dependencies as deps

    reset_singletons()
    monkeypatch.setenv("NEURO_API_STORE_ROOT", str(tmp_path / "exps"))
    monkeypatch.setenv("NEURO_API_ENABLE_VLM", "0")
    reset_singletons()
    store = ExperimentStore(root=tmp_path / "exps")
    svc = AnalysisService(store=store, mock_runner=runner, enable_vlm=False)
    svc.ensure_demo_experiment()

    def _svc():
        return svc

    def _store():
        return store

    monkeypatch.setattr(deps, "get_service", _svc)
    monkeypatch.setattr(deps, "get_store", _store)
    app = create_app()
    client = TestClient(app)
    return client, svc, store


def test_health(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "fastapi"
    assert "textModel" in body or "text_model" in body


def test_system_metrics(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.get("/api/system/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["precision"] == "INT8 W8A8"
    assert "visionModel" in body
    # unmeasured latency must be null, not invented
    assert body.get("tokensPerSec") is None
    assert body.get("p95LatencyMs") is None


def test_upload_empty_file(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.post(
        "/api/upload",
        data={"fileType": "figure", "filename": "x.png"},
        files={"file": ("x.png", b"", "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "empty_file"


def test_upload_missing_filename(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.post(
        "/api/upload",
        data={"fileType": "figure", "filename": "   "},
        files={"file": ("ignored.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "missing_filename"


def test_upload_unsupported_eeg_raw(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.post(
        "/api/upload",
        data={"fileType": "eeg", "filename": "raw.edf"},
        files={"file": ("raw.edf", b"not-an-edf", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "unsupported_file"


def test_upload_malformed_json(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.post(
        "/api/upload",
        data={"fileType": "metadata", "filename": "m.json"},
        files={"file": ("m.json", b"{not-json", "application/json")},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "malformed_input"


def test_upload_sample_metadata(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    payload = json.dumps({"sample_id": "S001_R01_E000"}).encode()
    r = client.post(
        "/api/upload",
        data={"fileType": "metadata", "filename": "sample.json"},
        files={"file": ("sample.json", payload, "application/json")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["experimentId"]
    assert "eeg" in body["detected_input_types"] or "metadata" in body["detected_input_types"]
    assert body["status"] == "ready"


def test_upload_figure(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    # minimal PNG header-ish bytes
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    r = client.post(
        "/api/upload",
        data={"fileType": "figure", "filename": "plot.png"},
        files={"file": ("plot.png", png, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["assetId"]
    assert "vision" in r.json()["detected_input_types"]


def test_invalid_experiment_id(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.get("/api/experiment/exp_does_not_exist")
    assert r.status_code == 404
    assert r.json()["error"] == "invalid_experiment_id"


def test_analyze_missing_question(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.post(
        "/api/analyze",
        json={"experimentId": "exp_demo_s001", "question": "   "},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "missing_question"


def test_analyze_schema_text_only(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.post(
        "/api/analyze",
        json={"experimentId": "exp_demo_s001", "question": "Which channel has highest beta?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "TEXT"
    assert isinstance(body["computed_evidence"], list)
    assert body["computed_evidence"], "expected numeric evidence"
    assert body["verification"]["status"] == "skipped"
    assert "totalMs" in body["timing"]
    assert body["system"]["precision"] == "INT8 W8A8"
    assert body["route_detail"]["requires_vision"] is False
    assert body["tools_used"]


def test_analyze_vision_required_mocked(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    # ensure demo has linked visuals from sample
    demo = svc.ensure_demo_experiment()
    assert demo.linked_image_ids or demo.visualizations
    r = client.post(
        "/api/analyze",
        json={
            "experiment_id": "exp_demo_s001",
            "question": "Visually describe the topomap for this sample",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "VISION"
    assert body["route_detail"]["requires_vision"] is True
    assert body["verification"]["status"] == "recovered"
    assert body["verification"]["recoveryPerformed"] is True
    assert body["verification"]["triggered"] is True
    # visual evidence from sample index (no VLM when enable_vlm=False)
    assert isinstance(body["visual_evidence"], list)


def test_analyze_verifier_recovery_shape(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockVerifierRunner())
    r = client.post(
        "/api/analyze",
        json={"experimentId": "exp_demo_s001", "question": "What is C3 beta power?"},
    )
    assert r.status_code == 200
    v = r.json()["verification"]
    assert v["triggered"] is True
    assert v["recovery_triggered"] is True
    assert v["status"] == "recovered"
    assert r.json()["timing"].get("recoveryMs") is not None or r.json()["timing"].get("recovery_ms") is not None


def test_analyze_invalid_experiment(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.post(
        "/api/analyze",
        json={"experimentId": "exp_missing", "question": "hello"},
    )
    assert r.status_code == 404


def test_analyze_vision_missing_image(tmp_path, monkeypatch):
    from neuro_agent.api import dependencies as deps

    reset_singletons()
    monkeypatch.setenv("NEURO_API_STORE_ROOT", str(tmp_path / "exps2"))
    reset_singletons()
    store = ExperimentStore(root=tmp_path / "exps2")
    svc = AnalysisService(store=store, mock_runner=MockVisionRunner(), enable_vlm=False)
    empty = store.create()
    monkeypatch.setattr(deps, "get_service", lambda: svc)
    monkeypatch.setattr(deps, "get_store", lambda: store)
    client = TestClient(create_app())
    r = client.post(
        "/api/analyze",
        json={"experimentId": empty.id, "question": "Describe the topomap"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "missing_image_for_vision"


def test_get_demo_experiment(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.get("/api/experiment/demo")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "exp_demo_s001"
    assert body.get("isDemo") is True
    assert body["visualizations"]
