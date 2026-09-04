#!/usr/bin/env python3
"""Stage H.1 systems benchmark for text BF16 / INT8 / INT4."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.evaluation.benchmark import (
    aggregate_benchmark,
    build_prompt,
    run_benchmark_iterations,
)
from neuro_agent.evaluation.schema import BenchmarkRunSummary
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.model_loader import load_model_and_tokenizer
from neuro_agent.paths import (
    BENCHMARKS_DIR,
    CONFIGS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    configure_hf_cache,
    ensure_dirs,
)
from neuro_agent.profiling.hardware import verify_hardware
from neuro_agent.quantization import normalize_quantization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text quantization systems benchmark")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="eval_quant_text_{bf16,int8,int4}.yaml (reuses model + bench block)",
    )
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--prompt-tokens", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()

    cfg = load_yaml(args.config)
    bench_defaults = load_yaml(CONFIGS_DIR / "benchmark.yaml")
    model_cfg = cfg["model"]
    inf_cfg = cfg["inference"]
    bench_cfg = {**bench_defaults.get("benchmark", {}), **cfg.get("benchmark", {})}
    prompt_cfg = bench_defaults.get("prompt", cfg.get("prompt", {}))

    warmup = args.warmup or int(bench_cfg.get("warmup_iterations", 2))
    iterations = args.iterations or int(bench_cfg.get("benchmark_iterations", 5))
    prompt_len = args.prompt_tokens or int(bench_cfg.get("prompt_token_length", 512))
    max_new = args.max_new_tokens or int(bench_cfg.get("max_new_tokens", 64))

    quant = normalize_quantization(model_cfg.get("quantization", "none")).value
    adapter_path = model_cfg.get("adapter_path")
    if adapter_path:
        adapter_path = str(PROJECT_ROOT / adapter_path)

    hw = verify_hardware()
    hardware = {
        "gpu": hw.gpus[0] if hw.gpus else {},
        "cuda_version": hw.cuda_version,
        "driver_version": hw.driver_version,
    }

    config = InferenceConfig(
        model_name=model_cfg["name"],
        dtype=model_cfg["dtype"],
        seed=inf_cfg["seed"],
        do_sample=False,
        max_new_tokens=max_new,
        use_cache=True,
        temperature=0.0,
        top_p=1.0,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        adapter_path=adapter_path,
        quantization=quant,
    )

    print(f"Loading {config.model_name} quant={quant} adapter={adapter_path}...")
    model, tokenizer, model_info = load_model_and_tokenizer(config)
    print(
        f"Loaded in {model_info.load_time_s:.2f}s, "
        f"weights={model_info.weight_memory_mb:.1f} MB, "
        f"alloc={model_info.allocated_after_load_mb}, "
        f"smi={model_info.nvidia_smi_mb}"
    )

    base_text = prompt_cfg.get(
        "base_text",
        "You are a neuroscience research assistant. Summarize EEG band power.",
    )
    prompt, actual_len = build_prompt(tokenizer, base_text, prompt_len)
    print(f"Prompt tokens: {actual_len}")

    raw, gpu_util = run_benchmark_iterations(
        model,
        tokenizer,
        config,
        prompt,
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
    )

    result = aggregate_benchmark(
        model_info,
        config,
        raw,
        experiment=f"text_quant_{quant}",
        context_length=actual_len,
        gpu_util=gpu_util,
        hardware=hardware,
        metadata={
            "model_variant": model_cfg.get("variant", f"sft_corrected_v2_{quant}"),
            "quantization": quant,
            "adapter_path": adapter_path,
            "warmup_iterations": warmup,
            "benchmark_iterations": iterations,
            "allocated_after_load_mb": model_info.allocated_after_load_mb,
            "peak_reserved_after_load_mb": model_info.peak_reserved_mb,
            "nvidia_smi_after_load_mb": model_info.nvidia_smi_mb,
        },
    )

    out_dir = RESULTS_DIR / "quantization" / "text" / "systems"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"systems_{quant}_{ts}.json"
    latest = out_dir / f"latest_{quant}.json"
    payload = result.to_dict()
    out_path.write_text(json.dumps(payload, indent=2))
    latest.write_text(json.dumps(payload, indent=2))

    # Also keep a copy under benchmarks/ without overwriting prior baseline artifacts
    bench_copy = BENCHMARKS_DIR / "quantization_text"
    bench_copy.mkdir(parents=True, exist_ok=True)
    (bench_copy / f"systems_{quant}_{ts}.json").write_text(json.dumps(payload, indent=2))

    summary = BenchmarkRunSummary(config_snapshot=cfg, results=[result])
    summary.save(out_dir / f"run_summary_{quant}_{ts}.json")

    print(f"Saved {out_path}")
    print(
        json.dumps(
            {
                "quantization": quant,
                "load_time_s": result.load_time_s,
                "weight_memory_mb": result.weight_memory_mb,
                "peak_vram_mb": result.peak_vram_mb,
                "ttft_ms_mean": result.ttft_ms.mean if result.ttft_ms else None,
                "decode_tps_mean": (
                    result.decode_tokens_per_second.mean
                    if result.decode_tokens_per_second
                    else None
                ),
                "e2e_ms_mean": (
                    result.end_to_end_latency_ms.mean if result.end_to_end_latency_ms else None
                ),
            },
            indent=2,
        )
    )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
