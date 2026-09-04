#!/usr/bin/env python3
"""H.8 targeted gate: BF16 vs H.7 W8A8 vs new balanced W8A8."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.evaluation.llm_eval import (
    load_eval_examples,
    verify_heldout_integrity,
    _build_prompt,
)
from neuro_agent.evaluation.verifiers import verify_example
from neuro_agent.paths import PROJECT_ROOT, configure_hf_cache

CONFIG = PROJECT_ROOT / "configs/eval_quant_text_bf16.yaml"
H7_PRED = PROJECT_ROOT / "results/quantization/w8a8_int8/full_quality_predictions.jsonl"
BF16_PRED = PROJECT_ROOT / "results/quantization/text/quality/bf16/predictions.jsonl"
CKPT = PROJECT_ROOT / "checkpoints/text_w8a8_int8_balanced_calibration"
OUT = PROJECT_ROOT / "results/quantization/w8a8_int8_quality_repair/targeted_gate.json"
PRED = PROJECT_ROOT / "results/quantization/w8a8_int8_quality_repair/targeted_predictions.jsonl"
FAMILIES = [
    "execution_vs_imagery",
    "movement_task_classification",
    "band_power_analysis",
    "channel_ranking",
    "numerical_reasoning",
    "statistical_comparison",
    "tool_selection",
    "factual_grounding",
]


def _load_pred(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "passed" not in rec and isinstance(rec.get("verification"), dict):
                rec["passed"] = rec["verification"].get("passed")
            out[rec["id"]] = rec
    return out


def _rates(ids: list[str], pred: dict[str, dict]) -> dict:
    by = defaultdict(lambda: {"n": 0, "pass": 0})
    for i in ids:
        rec = pred.get(i)
        if not rec:
            continue
        cat = rec["category"]
        by[cat]["n"] += 1
        by[cat]["pass"] += int(bool(rec.get("passed")))
    overall_n = sum(v["n"] for v in by.values())
    overall_p = sum(v["pass"] for v in by.values())
    return {
        "n": overall_n,
        "pass_rate": overall_p / overall_n if overall_n else None,
        "per_task": {
            k: (v["pass"] / v["n"] if v["n"] else None) for k, v in sorted(by.items())
        },
        "missing_ids": sum(1 for i in ids if i not in pred),
    }


def main() -> None:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    configure_hf_cache()
    cfg = load_yaml(CONFIG)
    eval_cfg = cfg["evaluation"]
    prompt_cfg = cfg["prompt"]
    inf_cfg = cfg["inference"]
    examples = load_eval_examples(PROJECT_ROOT / eval_cfg["dataset"])
    held_out = set(eval_cfg["held_out_subjects"])
    forbidden = set(eval_cfg.get("train_subjects", [])) | set(
        eval_cfg.get("validation_subjects", [])
    )
    verify_heldout_integrity(examples, held_out, forbidden)

    buckets: dict[str, list] = defaultdict(list)
    for ex in examples:
        buckets[ex["category"]].append(ex)
    subset = []
    subset.extend(buckets["execution_vs_imagery"][:20])
    for cat in FAMILIES:
        if cat == "execution_vs_imagery":
            continue
        subset.extend(buckets[cat][:10])
    ids = [ex["id"] for ex in subset]

    h7 = _load_pred(H7_PRED)
    bf16 = _load_pred(BF16_PRED)
    a = _rates(ids, bf16)
    b = _rates(ids, h7)

    tokenizer = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=False)
    llm = LLM(
        model=str(CKPT),
        quantization=None,
        enable_lora=False,
        dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
        trust_remote_code=False,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=int(inf_cfg["max_new_tokens"]))
    h8_pred = {}
    with PRED.open("w") as pred_f:
        for ex in subset:
            prompt = _build_prompt(ex, prompt_cfg["system"], tokenizer)
            outs = llm.generate([prompt], sampling_params=sampling)
            text = outs[0].outputs[0].text.strip()
            ver = verify_example(ex, text)
            rec = {
                "id": ex["id"],
                "category": ex["category"],
                "passed": ver.passed,
                "parse_error": ver.parse_error,
                "reason": ver.reason,
                "response": text,
            }
            pred_f.write(json.dumps(rec) + "\n")
            h8_pred[ex["id"]] = rec
    c = _rates(ids, h8_pred)

    exec_b = b["per_task"].get("execution_vs_imagery") or 0.0
    exec_c = c["per_task"].get("execution_vs_imagery") or 0.0
    other_regressed = []
    for cat in FAMILIES:
        if cat == "execution_vs_imagery":
            continue
        rb, rc = b["per_task"].get(cat), c["per_task"].get(cat)
        if rb is not None and rc is not None and (rb - rc) >= 0.15:
            other_regressed.append({"task": cat, "h7": rb, "h8": rc})

    improved = exec_c > 0.784 and (exec_c - exec_b) >= 0.05
    material = exec_c >= 0.88
    passed = improved and material and not other_regressed

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "subset": {
            "n": len(subset),
            "execution_vs_imagery": 20,
            "other_per_family": 10,
            "ids": ids,
        },
        "A_bf16": a,
        "B_h7_w8a8": b,
        "C_h8_balanced_w8a8": c,
        "execution_vs_imagery": {
            "bf16_full_ref": 1.0,
            "h7_full": 0.784,
            "h7_targeted": exec_b,
            "h8_targeted": exec_c,
            "delta_vs_h7_targeted": round(exec_c - exec_b, 4),
        },
        "other_family_regressions_ge_0.15": other_regressed,
        "gate": {
            "improved_above_0.784": exec_c > 0.784,
            "delta_ge_0.05_vs_h7_targeted": (exec_c - exec_b) >= 0.05,
            "material_recovery_ge_0.88": material,
            "no_major_other_regression": not other_regressed,
            "passed": passed,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "subset"}, indent=2))
    if passed:
        print("H8_TARGETED_PASS")
        sys.exit(0)
    print("H8_TARGETED_FAIL")
    sys.exit(2)


if __name__ == "__main__":
    main()
