"""Product component-routing evaluation (no model retraining).

Dataset covers conversation, concept, computation, vision, missing inputs,
follow-ups, and ambiguous requests. Metrics: component selection accuracy and
false-positive tool/vision/verifier rates.
"""

from __future__ import annotations

import json
from pathlib import Path

from neuro_agent.agent.task_plan import ArtifactContext, ConversationTurn, plan_task

EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "evals" / "component_routing_eval.json"


def _load_cases() -> list[dict]:
    return json.loads(EVAL_PATH.read_text())["cases"]


def _hist(raw: list[dict] | None) -> list[ConversationTurn]:
    out: list[ConversationTurn] = []
    for item in raw or []:
        out.append(
            ConversationTurn(
                role=item.get("role", "user"),
                content=item.get("content", ""),
                tools_used=list(item.get("tools_used") or []),
                route=item.get("route"),
                evidence_summary=item.get("evidence_summary"),
            )
        )
    return out


def test_component_routing_eval_metrics():
    cases = _load_cases()
    assert len(cases) >= 10

    correct = 0
    fp_tools = 0
    fp_vision = 0
    fp_verify = 0
    miss_input_ok = 0
    miss_input_n = 0
    follow_ok = 0
    follow_n = 0

    details = []
    for case in cases:
        arts = ArtifactContext(
            has_sample=bool(case.get("has_sample")),
            has_image=bool(case.get("has_image")),
            sample_id=case.get("sample_id"),
            image_id=case.get("image_id"),
        )
        plan = plan_task(
            case["question"],
            history=_hist(case.get("history")),
            artifacts=arts,
        )
        expected = set(case["expected_components"])
        got = set(plan.components)
        # Compare core execution flags rather than exact set equality for VERIFY optional
        core_exp = {c for c in expected if c != "VERIFY"}
        core_got = {c for c in got if c != "VERIFY"}
        match = core_exp == core_got
        if match:
            correct += 1
        else:
            details.append({"id": case["id"], "expected": sorted(core_exp), "got": sorted(core_got)})

        if "TOOLS" in got and "TOOLS" not in expected:
            fp_tools += 1
        if "VISION" in got and "VISION" not in expected:
            fp_vision += 1
        if "VERIFY" in got and "VERIFY" not in expected and "TOOLS" not in expected:
            fp_verify += 1

        if "NEEDS_INPUT" in expected:
            miss_input_n += 1
            if plan.needs_input:
                miss_input_ok += 1

        if case.get("expect_follow_up"):
            follow_n += 1
            if plan.is_follow_up:
                follow_ok += 1

    n = len(cases)
    accuracy = correct / n
    assert accuracy >= 0.85, f"component accuracy {accuracy:.2f}; misses={details}"
    assert fp_tools == 0, f"false-positive tools={fp_tools}"
    assert fp_vision == 0, f"false-positive vision={fp_vision}"
    if miss_input_n:
        assert miss_input_ok / miss_input_n >= 0.9
    if follow_n:
        assert follow_ok / follow_n >= 0.8

    # Persist a small metrics snapshot for the report (local only).
    metrics = {
        "n": n,
        "component_selection_accuracy": round(accuracy, 3),
        "false_positive_tools": fp_tools,
        "false_positive_vision": fp_vision,
        "false_positive_verifier": fp_verify,
        "missing_input_accuracy": round(miss_input_ok / miss_input_n, 3) if miss_input_n else None,
        "follow_up_resolution_accuracy": round(follow_ok / follow_n, 3) if follow_n else None,
        "misses": details,
    }
    out = Path(__file__).resolve().parents[1] / "results" / "component_routing_eval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
