"""
FINAL PRODUCTION ACCEPTANCE GATE — adversarial self-generated suite.

Expectations are locked BEFORE execution. Do not weaken after seeing results.
Run: .venv/bin/pytest tests/test_acceptance_gate.py -q --tb=line
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neuro_agent.agent.task_plan import ArtifactContext, ConversationTurn, plan_task
from neuro_agent.api.app import create_app
from neuro_agent.api.dependencies import reset_singletons
from neuro_agent.api.experiment_store import ExperimentStore
from neuro_agent.api.service import AnalysisService
from neuro_agent.paths import PROJECT_ROOT
from test_api_backend import MockTextRunner, MockVisionRunner, _wire_client

RESULTS = PROJECT_ROOT / "results" / "acceptance_gate_report.json"

# ---------- LOCKED EXPECTATIONS (do not edit after runs) ----------

NON_VISION_CASES = [
    {
        "id": "NV1",
        "prompt": "hey",
        "arts": {"has_sample": True, "has_image": True},
        "expect": {"text_only": True, "use_tools": False, "use_vision": False},
    },
    {
        "id": "NV2",
        "prompt": "what can you do?",
        "arts": {"has_sample": True, "has_image": True},
        "expect": {"text_only": True, "use_tools": False, "use_vision": False},
    },
    {
        "id": "NV3",
        "prompt": "What is EEG?",
        "arts": {"has_sample": True, "has_image": True},
        "expect": {"text_only": True, "use_tools": False, "use_vision": False},
    },
    {
        "id": "NV4",
        "prompt": "Explain event-related desynchronization in motor imagery.",
        "arts": {"has_sample": True, "has_image": True},
        "expect": {"text_only": True, "use_tools": False, "use_vision": False},
    },
    {
        "id": "NV5",
        "prompt": "What is C3 beta-band power for this sample?",
        "arts": {"has_sample": True, "has_image": False, "sample_id": "S026_R12_E000"},
        "expect": {"use_tools": True, "use_vision": False, "needs_input": False},
    },
    {
        "id": "NV6",
        "prompt": "Which five EEG channels have the highest beta-band power for this sample?",
        "arts": {"has_sample": True, "sample_id": "S026_R12_E000"},
        "expect": {"use_tools": True, "use_vision": False},
    },
    {
        "id": "NV7",
        "prompt": "Compare left_fist vs right_fist beta power for this sample",
        "arts": {"has_sample": True, "sample_id": "S026_R12_E000"},
        "expect": {"use_tools": True, "use_vision": False},
    },
    {
        "id": "NV8",
        "prompt": "Which five channels have the highest beta-band power for this sample?",
        "arts": {"has_sample": False, "has_image": True},
        "expect": {"needs_input": True, "need_kind": "dataset"},
    },
    {
        "id": "NV9",
        "prompt": "Why is T8 highest?",
        "arts": {"has_sample": True},
        "history": [
            {"role": "user", "content": "Which five channels have highest beta for this sample?"},
            {
                "role": "assistant",
                "content": "T8, IZ, O2, P7, O1. T8 is highest.",
                "tools_used": ["rank_channels_for_sample"],
                "evidence_summary": "T8=278",
            },
        ],
        "expect": {"text_only": True, "use_tools": False, "use_vision": False, "is_follow_up": True},
    },
    {
        "id": "NV10",
        "prompt": "hey",
        "arts": {"has_sample": True, "has_image": True},
        "history": [
            {"role": "user", "content": "Rank beta channels for this sample"},
            {
                "role": "assistant",
                "content": "T8 leads…",
                "tools_used": ["rank_channels_for_sample"],
            },
            {"role": "user", "content": "What do you see?"},
            {"role": "assistant", "content": "Left focus", "route": "VISION"},
        ],
        "expect": {"text_only": True, "use_tools": False, "use_vision": False},
    },
]

# Distinct synthetic PNGs (content bytes differ) — ground truth locked pre-run
VISION_CASES = [
    {
        "id": "V1",
        "filename": "eeg_waveform_trace.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"WAVEFORM_TRACE_AAAA" + b"\x00" * 20,
        "ground_truth": "EEG-style waveform / time-series plot",
        "question": "What do you see?",
        "expect_route": "VISION",
    },
    {
        "id": "V2",
        "filename": "scalp_topomap_beta.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"TOPOMAP_SCALP_BBBB" + b"\x00" * 20,
        "ground_truth": "Scalp topomap / spatial distribution map",
        "question": "Describe the dominant spatial pattern",
        "expect_route": "VISION",
    },
    {
        "id": "V3",
        "filename": "spectrogram_tf.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"SPECTROGRAM_CCCC" + b"\x00" * 20,
        "ground_truth": "Time-frequency spectrogram-like plot",
        "question": "What stands out in the upper region?",
        "expect_route": "VISION",
    },
    {
        "id": "V4",
        "filename": "psd_curve.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"PSD_CURVE_PLOT_DD" + b"\x00" * 20,
        "ground_truth": "Power spectral density curve",
        "question": "Interpret this",
        "expect_route": "VISION",
    },
    {
        "id": "V5",
        "filename": "correlation_heatmap.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"HEATMAP_GRID_EEEE" + b"\x00" * 20,
        "ground_truth": "Heatmap / matrix visualization",
        "question": "Is there asymmetry between the hemispheres?",
        "expect_route": "VISION",
    },
    {
        "id": "V6",
        "filename": "scatter_features.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"SCATTER_POINTS_FF" + b"\x00" * 20,
        "ground_truth": "Scatter plot of points",
        "question": "What does this show?",
        "expect_route": "VISION",
    },
    {
        "id": "V7",
        "filename": "line_chart_erp.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"LINE_CHART_ERP_GG" + b"\x00" * 20,
        "ground_truth": "Multi-line chart / ERP-like curves",
        "question": "What about the brighter areas?",
        "expect_route": "VISION",
    },
    {
        "id": "V8",
        "filename": "condition_bars.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"BAR_COMPARISON_HH" + b"\x00" * 20,
        "ground_truth": "Bar / comparison chart",
        "question": "Can you make sense of the left panel?",
        "expect_route": "VISION",
    },
    {
        "id": "V9",
        "filename": "multipanel_neuro.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"MULTI_PANEL_FIG_II" + b"\x00" * 20,
        "ground_truth": "Multi-panel neuroscience figure",
        "question": "Describe the dominant pattern on the left side",
        "expect_route": "VISION",
    },
    {
        "id": "V10",
        "filename": "anatomical_slice.png",
        "bytes": b"\x89PNG\r\n\x1a\n" + b"ANATOMY_SLICE_JJJ" + b"\x00" * 20,
        "ground_truth": "Anatomical / brain slice style image",
        "question": "What is visible in the central region?",
        "expect_route": "VISION",
    },
]


def _hist(raw):
    out = []
    for item in raw or []:
        out.append(
            ConversationTurn(
                role=item["role"],
                content=item["content"],
                tools_used=list(item.get("tools_used") or []),
                route=item.get("route"),
                evidence_summary=item.get("evidence_summary"),
            )
        )
    return out


def test_acceptance_non_vision_10():
    results = []
    fails = []
    for case in NON_VISION_CASES:
        arts = ArtifactContext(**case["arts"])
        plan = plan_task(case["prompt"], history=_hist(case.get("history")), artifacts=arts)
        exp = case["expect"]
        ok = True
        reasons = []
        for k, v in exp.items():
            got = getattr(plan, k)
            if got != v:
                ok = False
                reasons.append(f"{k}: expected {v!r} got {got!r}")
        row = {
            "id": case["id"],
            "prompt": case["prompt"],
            "pass": ok,
            "components": plan.components,
            "reason": plan.reason,
            "failures": reasons,
        }
        results.append(row)
        if not ok:
            fails.append(row)
    assert not fails, json.dumps(fails, indent=2)


def test_acceptance_vision_10_provenance_and_isolation(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    monkeypatch.setattr(svc, "enable_vlm", True)

    seen_ids: dict[str, str] = {}
    results = []
    fails = []

    for case in VISION_CASES:
        # Locked expected semantic behavior (pre-execution): VISION + exact asset provenance
        def _vlm(q: str, path, _case=case):  # noqa: ANN001
            # Return content keyed to THIS file's ground truth — never prior EEG labels
            return f"Visible content consistent with {_case['ground_truth']}."

        monkeypatch.setattr(svc, "_run_vlm_on_path", _vlm)

        up = client.post(
            "/api/upload",
            data={"fileType": "figure", "filename": case["filename"]},
            files={"file": (case["filename"], case["bytes"], "image/png")},
        )
        assert up.status_code == 200, up.text
        body_up = up.json()
        asset_id = body_up["assetId"]
        exp_id = body_up["experimentId"]
        seen_ids[case["id"]] = asset_id

        # Adversarial prior tool history that must NOT leak
        r = client.post(
            "/api/analyze",
            json={
                "experimentId": exp_id,
                "question": case["question"],
                "imageId": asset_id,
                "conversationHistory": [
                    {
                        "role": "user",
                        "content": "Compare left_fist vs right_fist for this sample",
                    },
                    {
                        "role": "assistant",
                        "content": "left_fist=0.168 right_fist=0.145 channels T8 C3",
                        "tools_used": ["compare_conditions"],
                        "route": "TEXT",
                        "evidence_summary": "left_fist=0.168",
                    },
                ],
            },
        )
        ok = True
        reasons = []
        if r.status_code != 200:
            ok = False
            reasons.append(f"status {r.status_code}")
            body = {}
        else:
            body = r.json()
            if body.get("route") != case["expect_route"]:
                ok = False
                reasons.append(f"route {body.get('route')}")
            sid = body.get("sourceImageId") or body.get("source_image_id")
            sname = body.get("sourceImageName") or body.get("source_image_name")
            if sid != asset_id:
                ok = False
                reasons.append(f"sourceImageId {sid} != {asset_id}")
            if sname != case["filename"]:
                ok = False
                reasons.append(f"sourceImageName {sname}")
            if not (body.get("visionUsed") or body.get("vision_used")):
                ok = False
                reasons.append("visionUsed false")
            ans = body.get("answer") or ""
            for bad in ("left_fist", "right_fist", "T8", "raw_model_output", "Uncertainty: None"):
                if bad.lower() in ans.lower():
                    ok = False
                    reasons.append(f"contamination {bad}")
            if case["ground_truth"] not in ans:
                ok = False
                reasons.append("answer missing locked ground-truth tag")
            if body.get("tools_used"):
                ok = False
                reasons.append(f"tools_used {body.get('tools_used')}")
            if body.get("computed_evidence"):
                ok = False
                reasons.append("computed_evidence present")

        row = {
            "id": case["id"],
            "filename": case["filename"],
            "ground_truth": case["ground_truth"],
            "question": case["question"],
            "asset_id": asset_id,
            "pass": ok,
            "failures": reasons,
            "answer": (body.get("answer") or "")[:200] if body else "",
        }
        results.append(row)
        if not ok:
            fails.append(row)

    # Multi-image A/B/A on same experiment
    a = VISION_CASES[0]
    b = VISION_CASES[5]
    up_a = client.post(
        "/api/upload",
        data={"fileType": "figure", "filename": a["filename"]},
        files={"file": (a["filename"], a["bytes"], "image/png")},
    )
    exp = up_a.json()["experimentId"]
    id_a = up_a.json()["assetId"]
    up_b = client.post(
        "/api/upload",
        data={"fileType": "figure", "filename": b["filename"], "experiment_id": exp},
        files={"file": (b["filename"], b["bytes"], "image/png")},
    )
    id_b = up_b.json()["assetId"]

    def run(img_id, name, tag):
        monkeypatch.setattr(
            svc,
            "_run_vlm_on_path",
            lambda q, p, t=tag: f"Analyzing {t}",
        )
        rr = client.post(
            "/api/analyze",
            json={"experimentId": exp, "question": "What do you see?", "imageId": img_id},
        )
        assert rr.status_code == 200
        bb = rr.json()
        assert (bb.get("sourceImageId") or bb.get("source_image_id")) == img_id
        assert (bb.get("sourceImageName") or bb.get("source_image_name")) == name
        assert tag in (bb.get("answer") or "")
        return bb

    run(id_a, a["filename"], "TAG_A")
    run(id_b, b["filename"], "TAG_B")
    run(id_a, a["filename"], "TAG_A_AGAIN")

    # No selection + visual task → needs_input (use empty experiment)
    er = client.post("/api/experiment", json={})
    empty = er.json().get("experimentId") or er.json().get("id")
    miss = client.post(
        "/api/analyze",
        json={"experimentId": empty, "question": "What do you see?"},
    )
    assert miss.status_code == 200
    assert miss.json()["route_detail"]["needs_input"] is True

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "vision_cases": results,
                "multi_image": {"id_a": id_a, "id_b": id_b},
                "fails": fails,
            },
            indent=2,
        )
    )
    assert not fails, json.dumps(fails, indent=2)


def test_acceptance_cross_modal_contamination(tmp_path, monkeypatch):
    client, svc, _ = _wire_client(tmp_path, monkeypatch, MockVisionRunner())
    monkeypatch.setattr(svc, "enable_vlm", True)
    monkeypatch.setattr(
        svc,
        "_run_vlm_on_path",
        lambda q, p: "Clean visual description of the uploaded figure only.",
    )

    # Seq1: after tool-looking history, vision must be clean
    up = client.post(
        "/api/upload",
        data={"fileType": "figure", "filename": "clean.png"},
        files={"file": ("clean.png", b"\x89PNG\r\n\x1a\n" + b"CLEANIMG" + b"\x00" * 16, "image/png")},
    )
    asset = up.json()["assetId"]
    exp = up.json()["experimentId"]
    r = client.post(
        "/api/analyze",
        json={
            "experimentId": exp,
            "question": "What do you see?",
            "imageId": asset,
            "conversationHistory": [
                {"role": "user", "content": "Rank beta for this sample"},
                {
                    "role": "assistant",
                    "content": "T8 left_fist right_fist",
                    "tools_used": ["rank_channels_for_sample"],
                    "evidence_summary": "T8=1 left_fist=2",
                },
            ],
        },
    )
    ans = r.json().get("answer") or ""
    assert "left_fist" not in ans.lower()
    assert "T8" not in ans

    # Seq3: greeting after vision → TEXT
    r2 = client.post(
        "/api/analyze",
        json={
            "experimentId": exp,
            "question": "hey",
            "imageId": asset,
            "conversationHistory": [
                {"role": "user", "content": "What do you see?"},
                {"role": "assistant", "content": ans, "route": "VISION"},
            ],
        },
    )
    assert r2.json()["route"] == "TEXT"
    assert not (r2.json().get("visionUsed") or r2.json().get("vision_used"))
