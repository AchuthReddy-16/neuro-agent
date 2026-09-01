#!/usr/bin/env python3
"""Run BF16 baseline inference benchmark."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_benchmark_config
from neuro_agent.evaluation.benchmark import (
    aggregate_benchmark,
    build_prompt,
    run_benchmark_iterations,
)
from neuro_agent.evaluation.schema import BenchmarkRunSummary
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.model_loader import load_model_and_tokenizer
from neuro_agent.paths import BENCHMARKS_DIR, configure_hf_cache, ensure_dirs
from neuro_agent.profiling.hardware import verify_hardware


def main() -> None:
    configure_hf_cache()
    ensure_dirs()
    cfg = load_benchmark_config()

    hw = verify_hardware()
    hardware = {
        "gpu": hw.gpus[0] if hw.gpus else {},
        "cuda_version": hw.cuda_version,
        "driver_version": hw.driver_version,
    }

    model_cfg = cfg["model"]
    inf_cfg = cfg["inference"]
    bench_cfg = cfg["benchmark"]
    prompt_cfg = cfg["prompt"]

    config = InferenceConfig(
        model_name=model_cfg["name"],
        dtype=model_cfg["dtype"],
        seed=inf_cfg["seed"],
        do_sample=inf_cfg["do_sample"],
        max_new_tokens=inf_cfg["max_new_tokens"],
        use_cache=inf_cfg["use_cache"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    print(f"Loading {config.model_name} in {config.dtype}...")
    model, tokenizer, model_info = load_model_and_tokenizer(config)
    print(
        f"Loaded in {model_info.load_time_s:.2f}s, "
        f"weights={model_info.weight_memory_mb:.1f} MB, "
        f"params={model_info.num_parameters:,}"
    )

    token_length = bench_cfg["prompt_token_length"]
    prompt, actual_len = build_prompt(tokenizer, prompt_cfg["base_text"], token_length)
    print(f"Prompt tokens: {actual_len}")

    raw, gpu_util = run_benchmark_iterations(
        model,
        tokenizer,
        config,
        prompt,
        warmup_iterations=bench_cfg["warmup_iterations"],
        benchmark_iterations=bench_cfg["benchmark_iterations"],
    )

    result = aggregate_benchmark(
        model_info,
        config,
        raw,
        experiment="baseline_bf16",
        context_length=actual_len,
        gpu_util=gpu_util,
        hardware=hardware,
        metadata={
            "model_variant": model_cfg["variant"],
            "warmup_iterations": bench_cfg["warmup_iterations"],
            "benchmark_iterations": bench_cfg["benchmark_iterations"],
        },
    )

    out_dir = BENCHMARKS_DIR / "baseline"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = BenchmarkRunSummary(
        config_snapshot=cfg,
        results=[result],
    )
    out_path = summary.save(out_dir / f"baseline_bf16_{ts}.json")
    result.save(out_dir / "latest_baseline.json")

    print(f"Baseline results saved to {out_path}")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
