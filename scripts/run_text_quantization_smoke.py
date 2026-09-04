#!/usr/bin/env python3
"""Stage H.1 smoke test: load BF16 / INT8 / INT4 base+adapter and generate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.model_loader import load_model_and_tokenizer, query_nvidia_smi_mb
from neuro_agent.paths import PROJECT_ROOT, RESULTS_DIR, configure_hf_cache, ensure_dirs
from neuro_agent.quantization import expected_weight_vram_mb, method_limitations, normalize_quantization


SMOKE_PROMPTS = [
    "What EEG frequency band is associated with motor imagery mu rhythm?",
    "List three common motor tasks in BCI2000 motor imagery experiments.",
    "Does execution or imagery typically produce stronger sensorimotor ERD?",
    "Name a typical EEG channel over left motor cortex.",
    "Give a one-sentence definition of band power analysis for EEG.",
]


def _free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_variant(quant: str, adapter_path: str, model_name: str) -> dict:
    quant_m = normalize_quantization(quant)
    config = InferenceConfig(
        model_name=model_name,
        dtype="bfloat16",
        seed=42,
        do_sample=False,
        max_new_tokens=48,
        use_cache=True,
        temperature=0.0,
        top_p=1.0,
        adapter_path=adapter_path,
        quantization=quant_m.value,
    )

    result: dict = {
        "quantization": quant_m.value,
        "method": {
            "bf16": "Transformers AutoModelForCausalLM torch_dtype=bfloat16",
            "int8": "bitsandbytes BitsAndBytesConfig(load_in_8bit=True)",
            "int4": "bitsandbytes NF4 + double quant (load_in_4bit)",
        }.get(quant_m.value, quant_m.value),
        "limitations": method_limitations(quant_m),
        "expected_weight_vram_mb_approx": expected_weight_vram_mb(4_000_000_000, quant_m),
        "supported": False,
        "error": None,
    }

    try:
        _free_cuda()
        model, tokenizer, info = load_model_and_tokenizer(config)
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        outputs = []
        for prompt in SMOKE_PROMPTS:
            messages = [
                {
                    "role": "system",
                    "content": "Answer briefly and directly.",
                },
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = tokenizer.decode(out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            outputs.append({"prompt": prompt, "response": gen.strip()})

        peak_alloc = peak_res = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            idx = device.index if device.index is not None else 0
            peak_alloc = torch.cuda.max_memory_allocated(idx) / (1024 * 1024)
            peak_res = torch.cuda.max_memory_reserved(idx) / (1024 * 1024)

        empty = [o for o in outputs if not o["response"]]
        result.update(
            {
                "supported": len(empty) == 0,
                "load_time_s": info.load_time_s,
                "weight_memory_mb": info.weight_memory_mb,
                "allocated_after_load_mb": info.allocated_after_load_mb,
                "peak_allocated_mb": peak_alloc,
                "peak_reserved_mb": peak_res,
                "nvidia_smi_mb": query_nvidia_smi_mb(),
                "num_parameters": info.num_parameters,
                "adapter_loaded": True,
                "outputs": outputs,
                "empty_output_count": len(empty),
            }
        )
        del model
        del tokenizer
        _free_cuda()
    except Exception as exc:  # noqa: BLE001 — smoke must record unsupported paths
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["supported"] = False
        _free_cuda()

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text quantization load smoke test")
    p.add_argument(
        "--variants",
        nargs="+",
        default=["bf16", "int8", "int4"],
        help="Quantization variants to smoke-test",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    p.add_argument(
        "--adapter",
        default="checkpoints/sft_corrected_v2/final",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()

    out_dir = RESULTS_DIR / "quantization" / "text" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    adapter = str(PROJECT_ROOT / args.adapter)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter": adapter,
        "variants": {},
        "all_supported": True,
    }

    for quant in args.variants:
        print(f"\n=== Smoke: {quant} ===")
        t0 = time.perf_counter()
        variant = run_variant(quant, adapter, args.model)
        variant["wall_time_s"] = time.perf_counter() - t0
        report["variants"][quant] = variant
        if not variant["supported"]:
            report["all_supported"] = False
        status = "PASS" if variant["supported"] else "FAIL"
        print(
            f"{status} {quant}: load={variant.get('load_time_s')}s "
            f"vram_alloc={variant.get('allocated_after_load_mb')} "
            f"smi={variant.get('nvidia_smi_mb')} err={variant.get('error')}"
        )
        if variant.get("outputs"):
            print(f"  sample: {variant['outputs'][0]['response'][:120]!r}")

    out_path = out_dir / f"smoke_{ts}.json"
    latest = out_dir / "latest_smoke.json"
    out_path.write_text(json.dumps(report, indent=2))
    latest.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    if not report["all_supported"]:
        print("One or more variants failed smoke; see report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
