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



class _MockAgentCompat:
    """Compat shims for production AnalysisService task-plan paths."""

    def ask(self, question: str, *, request_id: str | None = None, enable_verification: bool | None = None, **kwargs: Any) -> Any:
        return self._ask_impl(question, request_id=request_id)

    def ask_text_only(
        self,
        question: str,
        *,
        request_id: str | None = None,
        prior_context: str | None = None,
        history_snippet: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return _FakeTrace(
            request_id=request_id or "req_text_only",
            original_question=question,
            parsed_intent={"requires_vision": False, "text_only": True},
            tool_invocations=[],
            evidence_bundle={"success": True, "tool_invocations": [], "numeric_evidence": {}},
            final_answer=f"Conversational reply for: {question[:120]}",
            verification_triggered=False,
        )

    def _ask_impl(self, question: str, *, request_id: str | None = None) -> Any:
        raise NotImplementedError


class MockTextRunner(_MockAgentCompat):
    def _ask_impl(self, question: str, *, request_id: str | None = None) -> Any:
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


class MockVisionRunner(_MockAgentCompat):
    def _ask_impl(self, question: str, *, request_id: str | None = None) -> Any:
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


class MockVerifierRunner(_MockAgentCompat):
    """Text path with verifier + recovery shape."""

    def _ask_impl(self, question: str, *, request_id: str | None = None) -> Any:
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
    assert body["precision"] == "BF16"
    assert "visionModel" in body
    # unmeasured latency must be null, not invented
    assert body.get("tokensPerSec") is None
    assert body.get("p95LatencyMs") is None


def test_health_reports_runtime_status(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body.get("textStatus") == "ready" or body.get("text_status") == "ready"
    assert body.get("visionEnabled") is False or body.get("vision_enabled") is False
    assert "W8A8" not in (body.get("textModel") or body.get("text_model") or "")
    assert svc.state.precision == "BF16"


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
    assert body["system"]["precision"] == "BF16"
    assert body["route_detail"]["requires_vision"] is False
    assert body["tools_used"]


class MockFalseVisionSidecarRunner(_MockAgentCompat):
    """Simulates the known bug: ranking intent with vision sidecar fields set."""

    def _ask_impl(self, question: str, *, request_id: str | None = None) -> Any:
        return _FakeTrace(
            request_id=request_id or "req_false_vision",
            original_question=question,
            parsed_intent={
                "question_type": "channel_ranking",
                "requires_vision": False,
                "include_vision_evidence": True,
                "requested_visual_type": "topomap",
                "image_id": "img_fake_topomap",
                "sample_id": "S001_R01_E000",
            },
            tool_invocations=[{"name": "rank_channels_for_sample"}],
            evidence_bundle={
                "success": True,
                "ranked_evidence": {"ranking": ["C3", "Cz", "C4"], "values": {"C3": 0.2}},
                "numeric_evidence": [],
                "vision_evidence": [],
            },
            final_answer="Most discriminative channels: C3, Cz, C4.",
        )


class MockWrongRequiresVisionRunner(_MockAgentCompat):
    """Model incorrectly sets requires_vision for a pure ranking question."""

    def _ask_impl(self, question: str, *, request_id: str | None = None) -> Any:
        return _FakeTrace(
            request_id=request_id or "req_wrong_rv",
            original_question=question,
            parsed_intent={
                "question_type": "channel_ranking",
                "requires_vision": True,
                "include_vision_evidence": True,
                "requested_visual_type": "topomap",
                "sample_id": "S001_R01_E000",
            },
            tool_invocations=[{"name": "rank_channels_for_sample"}],
            evidence_bundle={
                "success": True,
                "ranked_evidence": {"ranking": ["C3", "C4"], "values": {"C3": 0.3}},
                "vision_evidence": [],
            },
            final_answer="C3 and C4 are most discriminative.",
        )


def test_routing_discriminative_channels_stays_text(tmp_path, monkeypatch):
    """Regression: ranking question must not require vision merely because a topomap exists."""
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockFalseVisionSidecarRunner())
    demo = svc.ensure_demo_experiment()
    assert demo.linked_image_ids, "demo must have linked figures to reproduce the bug condition"
    r = client.post(
        "/api/analyze",
        json={
            "experimentId": "exp_demo_s001",
            "question": "Which channels are most discriminative?",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "TEXT"
    assert body["route_detail"]["requires_vision"] is False
    assert body["tools_used"]
    assert body["answer"]
    warnings = " ".join(body.get("uncertainty") or "").lower()
    assert "vlm" not in warnings
    assert "missing vision" not in warnings


def test_routing_ignores_false_requires_vision_on_ranking(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockWrongRequiresVisionRunner())
    r = client.post(
        "/api/analyze",
        json={
            "experimentId": "exp_demo_s001",
            "question": "Which channels are most discriminative?",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "TEXT"
    assert body["route_detail"]["requires_vision"] is False


def test_decide_requires_vision_unit():
    from neuro_agent.api.service import AnalysisService
    from neuro_agent.api.experiment_store import ExperimentStore
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        svc = AnalysisService(store=ExperimentStore(root=Path(d)))
        assert (
            svc.decide_requires_vision(
                question="Which channels are most discriminative?",
                raw_intent={
                    "requires_vision": False,
                    "include_vision_evidence": True,
                    "requested_visual_type": "topomap",
                    "question_type": "channel_ranking",
                },
                image_id=None,
                visualization_id=None,
                context=None,
            )
            is False
        )
        assert (
            svc.decide_requires_vision(
                question="What does this figure show visually?",
                raw_intent={"requires_vision": True, "question_type": "visual_inspection"},
                image_id="img_1",
                visualization_id=None,
                context=None,
            )
            is True
        )
        assert (
            svc.decide_requires_vision(
                question="rank channels by beta",
                raw_intent={
                    "requires_vision": True,
                    "include_vision_evidence": True,
                    "requested_visual_type": "topomap",
                    "question_type": "channel_ranking",
                },
                image_id=None,
                visualization_id=None,
                context=None,
            )
            is False
        )


def test_analyze_vision_required_mocked(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    # ensure demo has linked visuals from sample — client must select explicitly
    demo = svc.ensure_demo_experiment()
    assert demo.linked_image_ids or demo.visualizations
    image_id = (demo.linked_image_ids or [demo.visualizations[0]["id"]])[0]
    r = client.post(
        "/api/analyze",
        json={
            "experiment_id": "exp_demo_s001",
            "question": "Visually describe the topomap for this sample",
            "image_id": image_id,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == "VISION"
    assert body["route_detail"]["requires_vision"] is True
    # Vision-primary path synthesizes via VLM/text-only shell; verifier is gated off.
    assert body["verification"]["status"] in {"skipped", "recovered", "passed", "triggered"}
    assert isinstance(body["visual_evidence"], list)
    assert body["visual_evidence"]


def test_analyze_vision_does_not_silently_use_sample_images(tmp_path, monkeypatch):
    """Live API must not auto-attach built-in sample topomaps."""
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
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
    assert body["route_detail"]["needs_input"] is True
    assert body["route_detail"]["need_kind"] == "image"
    assert "image" in (body.get("answer") or "").lower()
    assert body.get("tools_used") == []


def test_analyze_verifier_recovery_shape(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockVerifierRunner())
    r = client.post(
        "/api/analyze",
        json={
            "experimentId": "exp_demo_s001",
            "question": "Compute C3 beta-band power for this sample",
        },
    )
    assert r.status_code == 200, r.text
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


def test_analyze_strips_raw_model_output_from_uncertainty(tmp_path, monkeypatch):
    class _Runner(MockTextRunner):
        def _ask_impl(self, question: str, *, request_id: str | None = None) -> Any:
            t = super()._ask_impl(question, request_id=request_id)
            t.warnings = [
                'raw_model_output={"requires_vision": false, "question_type": "channel_ranking"}',
            ]
            t.final_answer = (
                "The highest beta-power channel is C3 at 0.184.\nUncertainty: None"
            )
            return t

    client, _, _ = _wire_client(tmp_path, monkeypatch, _Runner())
    r = client.post(
        "/api/analyze",
        json={
            "experimentId": "exp_demo_s001",
            "question": "Which channel has the highest beta power?",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    unc = body.get("uncertainty") or ""
    assert "raw_model_output" not in unc
    assert "requires_vision" not in unc
    assert "Answer:" not in (body.get("answer") or "")
    assert "C3" in (body.get("answer") or "")


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
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route_detail"]["needs_input"] is True
    assert body["route_detail"]["need_kind"] == "image"



def test_get_demo_experiment(tmp_path, monkeypatch):
    client, _, _ = _wire_client(tmp_path, monkeypatch, MockTextRunner())
    r = client.get("/api/experiment/demo")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "exp_demo_s001"
    assert body.get("isDemo") is True
    assert body["visualizations"]
