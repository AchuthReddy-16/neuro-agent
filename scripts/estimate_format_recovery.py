#!/usr/bin/env python3
"""Offline estimate of format-recovery from existing SFT predictions."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.evaluation.verifiers import normalize_token, verify_example
from neuro_agent.paths import PROJECT_ROOT

CHANNEL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,3})\b")
LABEL_RE = re.compile(
    r"\b(delta_power|theta_power|alpha_mu_power|beta_power|gamma_power|mu_power|rest|movement)\b",
    re.I,
)


def extract_answer_only(response: str, verification_type: str) -> str:
    text = response.strip()
    if not text:
        return text

    if verification_type == "numeric":
        nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", text)
        return nums[0] if nums else text.split()[0]

    if verification_type == "ranking":
        channels = CHANNEL_RE.findall(text.upper())
        if channels:
            return ", ".join(channels[:5])
        return text.split(";")[0].replace("The grounded descending order is", "").strip()

    if verification_type == "categorical":
        label_match = LABEL_RE.search(text)
        if label_match:
            return label_match.group(1).lower()
        channel_match = CHANNEL_RE.search(text.upper())
        if channel_match:
            return channel_match.group(1)
        first = text.split()[0].strip(".,;:")
        return first

    return text.split(".")[0].split(";")[0].strip()


def main() -> None:
    sft_path = PROJECT_ROOT / "results/multimodal_sft_eval/predictions.jsonl"
    base_path = PROJECT_ROOT / "results/multimodal_base_eval/predictions.jsonl"
    out_dir = PROJECT_ROOT / "results/multimodal_format_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_rows = {json.loads(l)["id"]: json.loads(l) for l in sft_path.open()}
    base_rows = {json.loads(l)["id"]: json.loads(l) for l in base_path.open()}

    recovered = 0
    total = 0
    per_verifier: dict[str, list[bool]] = defaultdict(list)
    per_task: dict[str, list[bool]] = defaultdict(list)

    for eid, row in sft_rows.items():
        total += 1
        vtype = row["verification_type"]
        trimmed = extract_answer_only(row["response"], vtype)
        example = {
            "verification_type": vtype,
            "ground_truth": row["ground_truth"],
            "context": {},
        }
        ver = verify_example(example, trimmed)
        per_verifier[vtype].append(ver.passed)
        per_task[row["category"]].append(ver.passed)
        if ver.passed:
            recovered += 1

    summary = {
        "method": "offline_answer_extraction_on_sft_outputs",
        "total_examples": total,
        "pass_rate": recovered / total,
        "base_pass_rate": sum(1 for r in base_rows.values() if r["verification"]["passed"]) / len(base_rows),
        "sft_pass_rate": sum(1 for r in sft_rows.values() if r["verification"]["passed"]) / len(sft_rows),
        "per_verifier": {
            k: {"pass_rate": sum(v) / len(v), "count": len(v)} for k, v in per_verifier.items()
        },
        "per_task": {
            k: {"pass_rate": sum(v) / len(v), "count": len(v)} for k, v in sorted(per_task.items())
        },
        "waveform_highest_rms_recovery": per_task.get("waveform_highest_rms", []),
    }
    if summary["waveform_highest_rms_recovery"]:
        wf = summary["waveform_highest_rms_recovery"]
        summary["waveform_highest_rms_pass_rate"] = sum(wf) / len(wf)

    with (out_dir / "offline_format_recovery.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
