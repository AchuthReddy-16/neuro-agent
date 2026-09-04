#!/usr/bin/env python3
"""H.7G — full 1000-example quality eval of vLLM W8A8 INT8 (same contract)."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
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

CKPT = PROJECT_ROOT / "checkpoints/text_w8a8_int8_compressed"
OUT = PROJECT_ROOT / "results/quantization/w8a8_int8/full_quality_eval.json"
PRED = PROJECT_ROOT / "results/quantization/w8a8_int8/full_quality_predictions.jsonl"
CONFIG = PROJECT_ROOT / "configs/eval_quant_text_bf16.yaml"


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
    integrity = verify_heldout_integrity(examples, held_out, forbidden)
    print(f"integrity n={integrity['example_count']} subjects={integrity['confirmed_subjects']}")

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
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(inf_cfg["max_new_tokens"]),
    )

    t0 = time.perf_counter()
    records = []
    invalid = 0
    passed = 0
    gen_tokens = 0
    per_task = defaultdict(lambda: {"n": 0, "pass": 0, "invalid": 0, "tokens": 0})

    with PRED.open("w") as pred_f:
        for i, ex in enumerate(examples, start=1):
            prompt = _build_prompt(ex, prompt_cfg["system"], tokenizer)
            outs = llm.generate([prompt], sampling_params=sampling)
            text = outs[0].outputs[0].text.strip()
            ntok = len(outs[0].outputs[0].token_ids)
            ver = verify_example(ex, text)
            rec = {
                "id": ex.get("id"),
                "category": ex.get("category"),
                "passed": ver.passed,
                "parse_error": ver.parse_error,
                "verification_type": ver.verification_type,
                "reason": ver.reason,
                "generated_tokens": ntok,
                "response": text,
            }
            pred_f.write(json.dumps(rec) + "\n")
            pred_f.flush()
            records.append(rec)
            gen_tokens += ntok
            if ver.passed:
                passed += 1
            if ver.parse_error:
                invalid += 1
            bucket = per_task[ex.get("category", "unknown")]
            bucket["n"] += 1
            bucket["pass"] += int(ver.passed)
            bucket["invalid"] += int(ver.parse_error)
            bucket["tokens"] += ntok
            if i % 50 == 0:
                print(f"  {i}/{len(examples)} pass={passed/i:.3f}", flush=True)

    runtime = time.perf_counter() - t0
    n = len(records)
    per_task_out = {
        k: {
            "n": v["n"],
            "verifier_pass_rate": v["pass"] / v["n"] if v["n"] else 0.0,
            "invalid_parse_rate": v["invalid"] / v["n"] if v["n"] else 0.0,
            "avg_generated_tokens": v["tokens"] / v["n"] if v["n"] else 0.0,
        }
        for k, v in sorted(per_task.items())
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "vllm_w8a8_int8",
        "checkpoint": str(CKPT),
        "n": n,
        "verifier_pass_rate": passed / n if n else 0.0,
        "invalid_parse_rate": invalid / n if n else 0.0,
        "avg_generated_tokens": gen_tokens / n if n else 0.0,
        "runtime_s": round(runtime, 1),
        "per_task": per_task_out,
        "reference": {"bf16_corrected_sft": 0.864, "bnb_int8": 0.866},
        "integrity": integrity,
        "task_distribution": dict(Counter(r["category"] for r in records)),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("H7G_OK")


if __name__ == "__main__":
    main()
