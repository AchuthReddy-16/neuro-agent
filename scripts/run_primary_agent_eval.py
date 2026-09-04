#!/usr/bin/env python3
"""Evaluate primary tool-using research agent (Stage G.3A)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from neuro_agent.agent.intent import intent_matches_expected
from neuro_agent.agent.research_agent import PrimaryResearchAgent, ResearchAgentConfig
from neuro_agent.agent.traces import AgentTrace
from neuro_agent.paths import PROJECT_ROOT, configure_hf_cache, ensure_dirs


TEST_SET = PROJECT_ROOT / "data" / "eval" / "agent_primary_test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "agent_primary_eval"

GATE = {
    "intent_schema_validity_min": 0.95,
    "tool_execution_success_min": 0.95,
    "unsupported_numeric_claims_max": 0,
    "max_tool_calls": 6,
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


def evaluate_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "intent_schema_validity": {
            "value": metrics["intent_schema_validity_rate"],
            "minimum": GATE["intent_schema_validity_min"],
            "passed": metrics["intent_schema_validity_rate"] >= GATE["intent_schema_validity_min"],
        },
        "tool_execution_success": {
            "value": metrics["tool_execution_success_rate"],
            "minimum": GATE["tool_execution_success_min"],
            "passed": metrics["tool_execution_success_rate"] >= GATE["tool_execution_success_min"],
        },
        "unsupported_numeric_claims": {
            "value": metrics["unsupported_numeric_claims"],
            "maximum": GATE["unsupported_numeric_claims_max"],
            "passed": metrics["unsupported_numeric_claims"] <= GATE["unsupported_numeric_claims_max"],
        },
        "all_intent_families": {
            "value": metrics["intent_families_covered"],
            "expected": 6,
            "passed": metrics["intent_families_covered"] == 6,
        },
        "max_tool_calls": {
            "value": metrics["max_tool_calls"],
            "maximum": GATE["max_tool_calls"],
            "passed": metrics["max_tool_calls"] <= GATE["max_tool_calls"],
        },
    }
    passed = all(c["passed"] for c in checks.values())
    return {"passed": passed, "checks": checks}


def run_evaluation(
    examples: list[dict[str, Any]],
    agent: PrimaryResearchAgent,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[AgentTrace] = []
    intent_families: set[str] = set()
    tool_calls_per_request: list[int] = []
    latencies: list[float] = []
    peak_vrams: list[float] = []
    failure_categories: Counter[str] = Counter()

    for ex in examples:
        trace = agent.ask(ex["question"])
        traces.append(trace)

        if trace.parsed_intent:
            intent_families.add(trace.parsed_intent.get("question_type", ""))
        tool_calls_per_request.append(len(trace.tool_invocations))
        latencies.append(trace.runtime_ms)
        peak_vrams.append(trace.peak_vram_mb)
        if trace.failure_category:
            failure_categories[trace.failure_category] += 1

    n = len(examples)
    intent_valid = sum(1 for t in traces if t.intent_valid)
    intent_accurate = 0
    for ex, trace in zip(examples, traces):
        from neuro_agent.agent.intent import IntentParseResult, parse_and_validate_intent

        if trace.intent_valid and trace.parsed_intent:
            parsed = IntentParseResult(
                success=True,
                raw_json=trace.parsed_intent,
                question_type=trace.parsed_intent.get("question_type"),
            )
            from neuro_agent.agent.intent import validate_intent

            try:
                parsed.request = validate_intent(trace.parsed_intent)
                if intent_matches_expected(parsed, ex["expected_intent"]):
                    intent_accurate += 1
            except Exception:
                pass

    tool_exec_success = sum(
        1 for t in traces if t.evidence_bundle and t.evidence_bundle.get("success")
    )
    e2e_success = sum(1 for t in traces if t.success)
    grounding_pass = sum(1 for t in traces if t.grounding and t.grounding.passed)
    unsupported_total = sum(
        len(t.grounding.unsupported_claims) for t in traces if t.grounding
    )

    metrics = {
        "total_examples": n,
        "intent_schema_validity_rate": _safe_rate(intent_valid, n),
        "intent_accuracy_rate": _safe_rate(intent_accurate, n),
        "valid_json_rate": _safe_rate(intent_valid, n),
        "tool_execution_success_rate": _safe_rate(tool_exec_success, n),
        "grounding_pass_rate": _safe_rate(grounding_pass, max(1, sum(1 for t in traces if t.grounding))),
        "e2e_success_rate": _safe_rate(e2e_success, n),
        "unsupported_numeric_claims": unsupported_total,
        "avg_tool_calls": sum(tool_calls_per_request) / n if n else 0.0,
        "max_tool_calls": max(tool_calls_per_request) if tool_calls_per_request else 0,
        "avg_latency_ms": sum(latencies) / n if n else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "peak_vram_mb": max(peak_vrams) if peak_vrams else 0.0,
        "intent_families_covered": len(intent_families),
        "intent_families": sorted(intent_families),
        "failure_categories": dict(failure_categories),
    }

    rows = []
    for ex, trace in zip(examples, traces):
        rows.append({"id": ex["id"], "question": ex["question"], **trace.to_dict()})

    return rows, metrics


def build_report(metrics: dict[str, Any], gate: dict[str, Any], traces: list[dict]) -> str:
    success_trace = next((t for t in traces if t.get("success")), None)
    failure_trace = next((t for t in traces if not t.get("success")), None)

    lines = [
        "# G.3A Primary Research Agent Evaluation Report",
        "",
        "## 1. Architecture",
        "User Question → PrimaryResearchAgent → Intent Selection (SFT model) → "
        "validate_intent → route_research_request → EvidenceBundle → "
        "Grounded Answer (SFT model).",
        "",
        "## 2. Model / Checkpoint",
        f"- Base: Qwen/Qwen3-4B-Instruct-2507",
        f"- Adapter: checkpoints/sft_corrected_v2/final",
        "",
        "## 3. Supported Intents",
        "band_power, rms, psd_peak, channel_ranking, threshold_set, condition_comparison",
        "",
        "## 4. Intent Parsing Success",
        f"- Schema validity: {metrics['intent_schema_validity_rate']:.1%}",
        f"- Intent accuracy vs expected: {metrics['intent_accuracy_rate']:.1%}",
        f"- Valid JSON rate: {metrics['valid_json_rate']:.1%}",
        "",
        "## 5. Tool Selection / Execution",
        f"- Tool execution success: {metrics['tool_execution_success_rate']:.1%}",
        f"- Intent families covered: {metrics['intent_families_covered']}/6 ({metrics['intent_families']})",
        "",
        "## 6. Avg Tool Calls",
        f"{metrics['avg_tool_calls']:.2f}",
        "",
        "## 7. Max Tool Calls",
        f"{metrics['max_tool_calls']}",
        "",
        "## 8. Grounding Results",
        f"- Grounding pass rate: {metrics['grounding_pass_rate']:.1%}",
        f"- Unsupported numeric claims: {metrics['unsupported_numeric_claims']}",
        "",
        "## 9. Latency",
        f"- Avg: {metrics['avg_latency_ms']:.1f} ms",
        f"- Max: {metrics['max_latency_ms']:.1f} ms",
        "",
        "## 10. Peak VRAM",
        f"{metrics['peak_vram_mb']:.1f} MB",
        "",
        "## 11. Success Trace Example",
    ]
    if success_trace:
        lines.append(f"ID: {success_trace['id']}")
        lines.append(f"Question: {success_trace['question']}")
        lines.append(f"Intent: {json.dumps(success_trace.get('parsed_intent'), indent=2)}")
        lines.append(f"Answer excerpt: {(success_trace.get('final_answer') or '')[:300]}")
    else:
        lines.append("None")

    lines.extend(["", "## 12. Failure Trace Example"])
    if failure_trace:
        lines.append(f"ID: {failure_trace['id']}")
        lines.append(f"Question: {failure_trace['question']}")
        lines.append(f"Category: {failure_trace.get('failure_category')}")
        lines.append(f"Errors: {failure_trace.get('errors')}")
    else:
        lines.append("None")

    lines.extend([
        "",
        "## 13. Gate Result",
        f"**{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "## 14. Blockers",
    ])
    blockers = [
        name for name, check in gate["checks"].items() if not check["passed"]
    ]
    lines.append(", ".join(blockers) if blockers else "None")

    lines.extend([
        "",
        "## 15. Ready for Verifier / Recovery?",
        "Yes" if gate["passed"] else "No — resolve blockers first.",
        "",
        f"E2E success rate: {metrics['e2e_success_rate']:.1%}",
        f"Failure categories: {json.dumps(metrics['failure_categories'])}",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G.3A primary agent evaluation")
    parser.add_argument("--test-set", type=Path, default=TEST_SET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_test_examples(args.test_set)
    if args.limit:
        examples = examples[: args.limit]

    config = ResearchAgentConfig()
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

    summary = {"metrics": metrics, "gate": gate}
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    report = build_report(metrics, gate, rows)
    with (args.output_dir / "report.md").open("w") as handle:
        handle.write(report)

    print(report)
    print(f"\nResults written to {args.output_dir}")
    if not gate["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
