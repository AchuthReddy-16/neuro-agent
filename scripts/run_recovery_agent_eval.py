#!/usr/bin/env python3
"""Evaluate conditional verifier + recovery agent (Stage G.3B)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.agent.intent import intent_matches_expected, validate_intent, IntentParseResult
from neuro_agent.agent.research_agent import PrimaryResearchAgent, ResearchAgentConfig
from neuro_agent.agent.traces import AgentTrace
from neuro_agent.paths import PROJECT_ROOT, configure_hf_cache, ensure_dirs

TEST_SET = PROJECT_ROOT / "data" / "eval" / "agent_recovery_test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "agent_recovery_eval"
TRACES_DIR = OUTPUT_DIR / "traces"

GATE = {
    "g3a_clean_e2e_min": 0.90,
    "unsupported_claims_max": 0,
    "verifier_trigger_rate_max": 0.50,
    "max_tool_calls": 6,
    "max_recovery_cycles": 1,
}


def load_test_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def build_corruption(spec: dict[str, Any] | None) -> Callable[[str], str] | None:
    if not spec:
        return None
    ctype = spec.get("type", "")

    if ctype == "corrupted_numeric":
        fake = spec.get("fake_value", 99999.0)

        def _corrupt(answer: str) -> str:
            return re.sub(
                r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
                str(fake),
                answer,
                count=1,
            )

        return _corrupt

    if ctype == "unsupported_channel":
        ch = spec.get("channel", "XX1")
        val = spec.get("value", 42.0)

        def _corrupt(answer: str) -> str:
            return answer + f"\nEvidence: Channel {ch} beta power is {val} µV²."

        return _corrupt

    if ctype == "unit_mismatch":

        def _corrupt(answer: str) -> str:
            return answer.replace("µV²", "Hz").replace("uV2", "Hz").replace("µV", "Hz")

        return _corrupt

    if ctype == "unit_mismatch_psd":

        def _corrupt(answer: str) -> str:
            return re.sub(r"(\d+(?:\.\d+)?)\s*Hz", r"\1 µV²", answer, count=1)

        return _corrupt

    if ctype == "missing_channel_claim":
        ch = spec.get("channel", "T8")

        def _corrupt(answer: str) -> str:
            return answer + f"\nEvidence: {ch} shows elevated beta power."

        return _corrupt

    if ctype == "wrong_condition":
        cond = spec.get("condition", "both_feet")

        def _corrupt(answer: str) -> str:
            return answer.replace("left_fist", cond).replace("right_fist", cond)

        return _corrupt

    if ctype == "wrong_metric_claim":

        def _corrupt(answer: str) -> str:
            return answer + "\nEvidence: Top channel ranked by gamma_power with value 999.0."

        return _corrupt

    if ctype == "missing_sections":

        def _corrupt(answer: str) -> str:
            return "The beta power is approximately 1.0."

        return _corrupt

    return None


def evaluate_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "g3a_clean_e2e": {
            "value": metrics["g3a_clean_e2e_rate"],
            "minimum": GATE["g3a_clean_e2e_min"],
            "passed": metrics["g3a_clean_e2e_rate"] >= GATE["g3a_clean_e2e_min"],
        },
        "unsupported_claims": {
            "value": metrics["unsupported_numeric_claims"],
            "maximum": GATE["unsupported_claims_max"],
            "passed": metrics["unsupported_numeric_claims"] <= GATE["unsupported_claims_max"],
        },
        "verifier_not_always_on": {
            "value": metrics["verifier_trigger_rate"],
            "maximum": GATE["verifier_trigger_rate_max"],
            "passed": metrics["verifier_trigger_rate"] <= GATE["verifier_trigger_rate_max"],
        },
        "max_tool_calls": {
            "value": metrics["max_tool_calls"],
            "maximum": GATE["max_tool_calls"],
            "passed": metrics["max_tool_calls"] <= GATE["max_tool_calls"],
        },
        "max_recovery_cycles": {
            "value": metrics["max_recovery_cycles_observed"],
            "maximum": GATE["max_recovery_cycles"],
            "passed": metrics["max_recovery_cycles_observed"] <= GATE["max_recovery_cycles"],
        },
        "recovery_improves": {
            "value": metrics["recovery_success_rate"],
            "minimum": 0.5,
            "passed": metrics["recovery_success_rate"] >= 0.5
            or metrics["recovery_attempts"] == 0,
        },
    }
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def run_evaluation(
    examples: list[dict[str, Any]],
    agent: PrimaryResearchAgent,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[AgentTrace] = []
    categories: Counter[str] = Counter()

    for ex in examples:
        corruption = build_corruption(ex.get("inject_corruption"))
        trace = agent.ask(ex["question"], draft_corruption=corruption)
        traces.append(trace)
        categories[ex.get("category", "unknown")] += 1

    n = len(examples)
    normal_examples = [ex for ex in examples if ex.get("category") == "normal_easy"]
    normal_ids = {ex["id"] for ex in normal_examples}
    normal_traces = [t for ex, t in zip(examples, traces) if ex["id"] in normal_ids]

    corruption_examples = [ex for ex in examples if ex.get("inject_corruption")]
    corruption_ids = {ex["id"] for ex in corruption_examples}
    corruption_traces = [t for ex, t in zip(examples, traces) if ex["id"] in corruption_ids]

    first_pass_accept = sum(
        1
        for t in traces
        if t.first_pass_verification and t.first_pass_verification.get("passed")
        and not t.recovery
    )
    verifier_triggered = sum(1 for t in traces if t.verification_triggered)
    deterministic_only = sum(
        1
        for t in traces
        if not t.verification_triggered and t.success and t.path_mode == "NORMAL"
    )
    verifier_model_calls = sum(
        1 for t in traces if t.verification_triggered and t.verifier_latency_ms > 0
    )
    recovery_attempts = sum(1 for t in traces if t.recovery is not None)
    recovery_success = sum(1 for t in traces if t.recovery and t.recovery.success)
    e2e_success = sum(1 for t in traces if t.success)
    g3a_clean_success = sum(1 for t in normal_traces if t.success)

    unsupported_total = sum(
        len(t.grounding.unsupported_claims) for t in traces if t.grounding and t.success is False
    )
    unsupported_total += sum(
        len(t.grounding.unsupported_claims)
        for t in traces
        if t.grounding and t.success and t.grounding.passed is False
    )

    false_rejections = sum(
        1
        for ex, t in zip(examples, traces)
        if ex.get("category") == "normal_easy"
        and not t.success
        and t.path_mode == "NORMAL"
    )

    normal_latencies = [t.runtime_ms for t in normal_traces if t.path_mode == "NORMAL"]
    recovery_latencies = [t.runtime_ms for t in traces if t.path_mode == "RECOVERY"]
    tool_calls = [len(t.tool_invocations) for t in traces]
    model_calls = [t.model_calls for t in traces]

    corruption_recovery_success = sum(
        1
        for ex, t in zip(examples, traces)
        if ex.get("expect_recovery") and t.recovery and t.success
    )
    corruption_recovery_expected = sum(1 for ex in examples if ex.get("expect_recovery"))

    metrics = {
        "total_examples": n,
        "categories": dict(categories),
        "first_pass_acceptance_rate": _safe_rate(first_pass_accept, n),
        "verifier_trigger_rate": _safe_rate(verifier_triggered, n),
        "deterministic_only_rate": _safe_rate(deterministic_only, n),
        "verifier_model_call_rate": _safe_rate(verifier_model_calls, n),
        "recovery_rate": _safe_rate(recovery_attempts, n),
        "recovery_attempts": recovery_attempts,
        "recovery_success_rate": _safe_rate(recovery_success, max(1, recovery_attempts)),
        "corruption_recovery_success_rate": _safe_rate(
            corruption_recovery_success, max(1, corruption_recovery_expected)
        ),
        "e2e_success_rate": _safe_rate(e2e_success, n),
        "g3a_clean_e2e_rate": _safe_rate(g3a_clean_success, max(1, len(normal_traces))),
        "unsupported_numeric_claims": unsupported_total,
        "false_rejection_count": false_rejections,
        "avg_tool_calls": sum(tool_calls) / n if n else 0.0,
        "max_tool_calls": max(tool_calls) if tool_calls else 0,
        "avg_model_calls": sum(model_calls) / n if n else 0.0,
        "max_model_calls": max(model_calls) if model_calls else 0,
        "avg_normal_latency_ms": sum(normal_latencies) / len(normal_latencies)
        if normal_latencies
        else 0.0,
        "avg_recovery_latency_ms": sum(recovery_latencies) / len(recovery_latencies)
        if recovery_latencies
        else 0.0,
        "max_recovery_cycles_observed": 1 if recovery_attempts else 0,
        "peak_vram_mb": max((t.peak_vram_mb for t in traces), default=0.0),
        "intent_accuracy_rate": _intent_accuracy(examples, traces),
    }
    return _rows(examples, traces), metrics


def _intent_accuracy(examples: list[dict[str, Any]], traces: list[AgentTrace]) -> float:
    correct = 0
    for ex, trace in zip(examples, traces):
        if not trace.intent_valid or not trace.parsed_intent:
            continue
        try:
            parsed = IntentParseResult(
                success=True,
                raw_json=trace.parsed_intent,
                question_type=trace.parsed_intent.get("question_type"),
            )
            parsed.request = validate_intent(trace.parsed_intent)
            if intent_matches_expected(parsed, ex.get("expected_intent", {})):
                correct += 1
        except Exception:
            pass
    return _safe_rate(correct, len(examples))


def _rows(examples: list[dict[str, Any]], traces: list[AgentTrace]) -> list[dict[str, Any]]:
    rows = []
    for ex, trace in zip(examples, traces):
        rows.append(
            {
                "id": ex["id"],
                "category": ex.get("category"),
                "question": ex["question"],
                "expect_verifier_trigger": ex.get("expect_verifier_trigger"),
                "expect_recovery": ex.get("expect_recovery"),
                **trace.to_dict(),
            }
        )
    return rows


def build_report(metrics: dict[str, Any], gate: dict[str, Any], traces: list[dict]) -> str:
    success_trace = next((t for t in traces if t.get("success")), None)
    recovery_trace = next((t for t in traces if t.get("recovery")), None)
    normal_trace = next((t for t in traces if t.get("category") == "normal_easy"), None)

    lines = [
        "# G.3B Conditional Verifier + Recovery Agent Evaluation Report",
        "",
        "## 1. Architecture",
        "User Question → PrimaryResearchAgent → Intent → Tools → Draft Answer → "
        "Deterministic Checks → (conditional) Verifier Model → (if fail) Recovery (max 1) → "
        "Re-verify → Final Answer.",
        "",
        "## 2. Model / Checkpoint",
        "- Base: Qwen/Qwen3-4B-Instruct-2507",
        "- Adapter: checkpoints/sft_corrected_v2/final",
        "",
        "## 3. Verification Schema",
        "VerificationResult: passed, confidence_score, failure_codes, unsupported_claims, "
        "evidence_conflicts, missing_evidence, unit_issues, condition_mismatch, recommendation.",
        "",
        "## 4. Deterministic Checks",
        "Numeric grounding, channel existence, units, conditions, tool success, "
        "required evidence, conflicting numerics, tool loops, answer sections.",
        "",
        "## 5. Trigger Policy",
        f"- Verifier trigger rate: {metrics['verifier_trigger_rate']:.1%}",
        f"- Deterministic-only acceptance: {metrics['deterministic_only_rate']:.1%}",
        "",
        "## 6. First-Pass Acceptance",
        f"{metrics['first_pass_acceptance_rate']:.1%}",
        "",
        "## 7. Verifier Model Call Rate",
        f"{metrics['verifier_model_call_rate']:.1%}",
        "",
        "## 8. Recovery Metrics",
        f"- Recovery rate: {metrics['recovery_rate']:.1%} ({metrics['recovery_attempts']} attempts)",
        f"- Recovery success: {metrics['recovery_success_rate']:.1%}",
        f"- Corruption recovery success: {metrics['corruption_recovery_success_rate']:.1%}",
        "",
        "## 9. E2E Success",
        f"- Overall: {metrics['e2e_success_rate']:.1%}",
        f"- G.3A clean subset: {metrics['g3a_clean_e2e_rate']:.1%}",
        "",
        "## 10. Unsupported Claims",
        f"{metrics['unsupported_numeric_claims']}",
        "",
        "## 11. False Rejections (normal_easy)",
        f"{metrics['false_rejection_count']}",
        "",
        "## 12. Tool / Model Calls",
        f"- Avg tool calls: {metrics['avg_tool_calls']:.2f}",
        f"- Max tool calls: {metrics['max_tool_calls']}",
        f"- Avg model calls: {metrics['avg_model_calls']:.2f}",
        "",
        "## 13. Latency (NORMAL vs RECOVERY)",
        f"- Avg NORMAL path: {metrics['avg_normal_latency_ms']:.1f} ms",
        f"- Avg RECOVERY path: {metrics['avg_recovery_latency_ms']:.1f} ms",
        "",
        "## 14. Peak VRAM",
        f"{metrics['peak_vram_mb']:.1f} MB",
        "",
        "## 15. Normal-Path Trace Example",
    ]
    if normal_trace:
        lines.append(f"ID: {normal_trace['id']}")
        lines.append(f"Verifier triggered: {normal_trace.get('verification_triggered')}")
        lines.append(f"Path: {normal_trace.get('path_mode')}")
        lines.append(f"Latency: {normal_trace.get('runtime_ms', 0):.1f} ms")
    else:
        lines.append("None")

    lines.extend(["", "## 16. Recovery Trace Example"])
    if recovery_trace:
        lines.append(f"ID: {recovery_trace['id']}")
        lines.append(f"Recovery action: {recovery_trace.get('recovery', {}).get('action')}")
        lines.append(f"Recovery success: {recovery_trace.get('recovery', {}).get('success')}")
    else:
        lines.append("None")

    lines.extend([
        "",
        "## 17. Success Trace Example",
    ])
    if success_trace:
        lines.append(f"ID: {success_trace['id']}")
        lines.append(f"Answer excerpt: {(success_trace.get('final_answer') or '')[:200]}")
    else:
        lines.append("None")

    lines.extend([
        "",
        "## 18. Categories",
        json.dumps(metrics.get("categories", {}), indent=2),
        "",
        "## 19. Gate Result",
        f"**{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "## 20. Blockers",
    ])
    blockers = [name for name, check in gate["checks"].items() if not check["passed"]]
    lines.append(", ".join(blockers) if blockers else "None")

    lines.extend([
        "",
        "## 21. Intent Accuracy",
        f"{metrics['intent_accuracy_rate']:.1%}",
        "",
        "## 22. Traces Location",
        f"results/agent_recovery_eval/traces/",
    ])
    return "\n".join(lines)


def _git_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception as exc:
        return f"(unavailable: {exc})"


def _git_diff_stat() -> str:
    try:
        return subprocess.check_output(
            ["git", "diff", "--stat"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception as exc:
        return f"(unavailable: {exc})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G.3B recovery agent evaluation")
    parser.add_argument("--test-set", type=Path, default=TEST_SET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    examples = load_test_examples(args.test_set)
    if args.limit:
        examples = examples[: args.limit]

    config = ResearchAgentConfig(enable_verification=True, max_recovery_cycles=1)
    agent = PrimaryResearchAgent(config)
    print(f"Loading model {config.model_name} + {config.adapter_path}...")
    agent.load()

    t0 = time.perf_counter()
    rows, metrics = run_evaluation(examples, agent)
    metrics["eval_runtime_s"] = time.perf_counter() - t0
    metrics["model"] = config.model_name
    metrics["adapter_path"] = config.adapter_path

    gate = evaluate_gate(metrics)

    with (args.output_dir / "traces.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    for row in rows:
        trace_path = TRACES_DIR / f"{row['id']}.json"
        with trace_path.open("w") as handle:
            json.dump(row, handle, indent=2, sort_keys=True)

    summary = {"metrics": metrics, "gate": gate}
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    report = build_report(metrics, gate, rows)
    report += "\n\n## Git Status\n```\n" + _git_status() + "\n```\n"
    report += "\n## Git Diff Stat\n```\n" + _git_diff_stat() + "\n```\n"

    with (args.output_dir / "report.md").open("w") as handle:
        handle.write(report)

    print(report)
    print(f"\nResults written to {args.output_dir}")
    if not gate["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
