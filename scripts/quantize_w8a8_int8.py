#!/usr/bin/env python3
"""H.7D — SmoothQuant + GPTQ W8A8 INT8 via llm-compressor (isolated env)."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED = PROJECT_ROOT / "checkpoints/text_merged_corrected_bf16"
OUT_DIR = Path(
    os.environ.get(
        "W8A8_OUT_DIR",
        str(PROJECT_ROOT / "checkpoints/text_w8a8_int8_compressed"),
    )
)
CALIB = Path(
    os.environ.get(
        "W8A8_CALIB",
        str(PROJECT_ROOT / "results/quantization/w8a8_int8/calibration_prompts.jsonl"),
    )
)
REPORT = Path(
    os.environ.get(
        "W8A8_REPORT",
        str(PROJECT_ROOT / "results/quantization/w8a8_int8/quantization_report.json"),
    )
)
LOG = PROJECT_ROOT / "results/quantization/w8a8_int8/quantization.log"

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SMOOTHQUANT_STRENGTH = 0.8
SCHEME = "W8A8"
IGNORE = ["lm_head"]
MAX_SEQ_LENGTH = 2048


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> None:
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    warnings: list[str] = []
    t0 = time.perf_counter()
    print(f"Loading merged BF16 from {MERGED}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MERGED), trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(MERGED),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )

    rows = []
    with CALIB.open() as handle:
        for line in handle:
            rec = json.loads(line)
            rows.append({"text": rec["text"]})
    ds = Dataset.from_list(rows)
    print(f"Calibration rows: {len(ds)}", flush=True)

    recipe = [
        SmoothQuantModifier(smoothing_strength=SMOOTHQUANT_STRENGTH),
        GPTQModifier(targets="Linear", scheme=SCHEME, ignore=IGNORE),
    ]

    print("Starting oneshot W8A8...", flush=True)
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=len(ds),
        pad_to_max_length=False,
        text_column="text",
        output_dir=str(OUT_DIR),
        save_compressed=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)
    elapsed = time.perf_counter() - t0

    cfg_path = OUT_DIR / "config.json"
    compression_config = None
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        compression_config = cfg.get("quantization_config") or cfg.get("compression_config")

    report = {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(MERGED),
        "output_checkpoint": str(OUT_DIR),
        "quantization_runtime_s": round(elapsed, 2),
        "disk_checkpoint_bytes": _dir_size_bytes(OUT_DIR),
        "disk_checkpoint_gb": round(_dir_size_bytes(OUT_DIR) / (1024**3), 4),
        "parameters": {
            "smoothquant_smoothing_strength": SMOOTHQUANT_STRENGTH,
            "smoothquant_alpha": SMOOTHQUANT_STRENGTH,
            "gptq_targets": "Linear",
            "gptq_scheme": SCHEME,
            "weight_quantization": "INT8 symmetric per-channel (W8A8 GPTQ)",
            "activation_quantization": "INT8 symmetric dynamic per-token",
            "per_channel_weights": True,
            "per_token_activations": True,
            "symmetric": True,
            "calibration_samples": len(ds),
            "max_seq_length": MAX_SEQ_LENGTH,
            "excluded_layers": IGNORE,
        },
        "compression_config": compression_config,
        "warnings": warnings,
        "fp8_used": False,
        "bitsandbytes_used": False,
        "int4_used": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in report if k != "compression_config"}, indent=2))
    print("QUANTIZE_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        tb = traceback.format_exc()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            json.dumps(
                {"status": "failed", "error": str(exc), "traceback": tb},
                indent=2,
            )
        )
        print(tb)
        sys.exit(1)
