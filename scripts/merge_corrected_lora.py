#!/usr/bin/env python3
"""H.7B — merge corrected LoRA into a standalone BF16 checkpoint."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neuro_agent.paths import PROJECT_ROOT, configure_hf_cache, ensure_dirs

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = PROJECT_ROOT / "checkpoints/sft_corrected_v2/final"
OUT_DIR = PROJECT_ROOT / "checkpoints/text_merged_corrected_bf16"


def main() -> None:
    configure_hf_cache()
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base {BASE_MODEL} BF16...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        trust_remote_code=False,
    )
    print(f"Applying adapter {ADAPTER}...")
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    print("merge_and_unload()...")
    model = model.merge_and_unload()
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUT_DIR)
    elapsed = time.perf_counter() - t0

    peft_left = type(model).__name__
    report = {
        "base_model": BASE_MODEL,
        "adapter": str(ADAPTER),
        "output": str(OUT_DIR),
        "merged_class": peft_left,
        "dtype": str(next(model.parameters()).dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(elapsed, 2),
        "peft_runtime_lora": False,
    }
    (OUT_DIR / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("MERGE_OK")


if __name__ == "__main__":
    main()
