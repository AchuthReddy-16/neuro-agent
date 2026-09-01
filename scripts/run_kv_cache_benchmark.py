#!/usr/bin/env python3
"""Run KV-cache experiments: use_cache on/off and context-length scaling."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from neuro_agent.config import load_benchmark_config
from neuro_agent.evaluation.benchmark import (
    aggregate_benchmark,
    build_prompt,
    run_benchmark_iterations,
)
from neuro_agent.evaluation.schema import BenchmarkRunSummary
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.engine import estimate_kv_cache_mb
from neuro_agent.inference.model_loader import load_model_and_tokenizer
from neuro_agent.paths import BENCHMARKS_DIR, configure_hf_cache, ensure_dirs
from neuro_agent.profiling.hardware import verify_hardware


def _safe_context_lengths(cfg: dict) -> list[int]:
    lengths = list(cfg["benchmark"]["context_lengths"])
    if cfg["benchmark"].get("try_context_length_8192"):
        lengths.append(8192)
    return lengths


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
    kv_cfg = bench_cfg["kv_cache_experiments"]
    prompt_cfg = cfg["prompt"]

    base_config = InferenceConfig(
        model_name=model_cfg["name"],
        dtype=model_cfg["dtype"],
        seed=inf_cfg["seed"],
        do_sample=inf_cfg["do_sample"],
        max_new_tokens=inf_cfg["max_new_tokens"],
        use_cache=True,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    print(f"Loading {base_config.model_name}...")
    model, tokenizer, model_info = load_model_and_tokenizer(base_config)
    print(f"Loaded: {model_info.weight_memory_mb:.1f} MB weights")

    all_results = []

    # --- A/B: use_cache True vs False ---
    for use_cache, label in [
        (True, "kv_cache_on"),
        (False, "kv_cache_off"),
    ]:
        if use_cache and not kv_cfg.get("run_use_cache_true", True):
            continue
        if not use_cache and not kv_cfg.get("run_use_cache_false", True):
            continue

        config = InferenceConfig(
            model_name=base_config.model_name,
            dtype=base_config.dtype,
            seed=base_config.seed,
            do_sample=base_config.do_sample,
            max_new_tokens=base_config.max_new_tokens,
            use_cache=use_cache,
            trust_remote_code=base_config.trust_remote_code,
        )

        token_length = bench_cfg["prompt_token_length"]
        prompt, actual_len = build_prompt(tokenizer, prompt_cfg["base_text"], token_length)

        print(f"\n=== {label} (prompt={actual_len} tokens, use_cache={use_cache}) ===")
        try:
            raw, gpu_util = run_benchmark_iterations(
                model,
                tokenizer,
                config,
                prompt,
                warmup_iterations=bench_cfg["warmup_iterations"],
                benchmark_iterations=bench_cfg["benchmark_iterations"],
            )
            kv_est = estimate_kv_cache_mb(model, actual_len + config.max_new_tokens)
            result = aggregate_benchmark(
                model_info,
                config,
                raw,
                experiment=label,
                context_length=actual_len,
                gpu_util=gpu_util,
                hardware=hardware,
                kv_cache_estimate_mb=kv_est,
                metadata={"model_variant": model_cfg["variant"]},
            )
            all_results.append(result)
            print(
                f"  prefill={result.prefill_latency_ms.mean:.1f}ms "
                f"ttft={result.ttft_ms.mean:.1f}ms "
                f"decode={result.decode_tokens_per_second.mean:.1f} tok/s "
                f"peak_vram={result.peak_vram_mb:.0f} MB"
            )
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  SKIPPED {label}: OOM — {exc}")
            torch.cuda.empty_cache()

    # --- C: Context-length scaling (use_cache=True) ---
    config = InferenceConfig(
        model_name=base_config.model_name,
        dtype=base_config.dtype,
        seed=base_config.seed,
        do_sample=base_config.do_sample,
        max_new_tokens=base_config.max_new_tokens,
        use_cache=True,
        trust_remote_code=base_config.trust_remote_code,
    )

    for ctx_len in _safe_context_lengths(cfg):
        prompt, actual_len = build_prompt(tokenizer, prompt_cfg["base_text"], ctx_len)
        print(f"\n=== context_scaling len={actual_len} ===")
        try:
            raw, gpu_util = run_benchmark_iterations(
                model,
                tokenizer,
                config,
                prompt,
                warmup_iterations=max(1, bench_cfg["warmup_iterations"] - 1),
                benchmark_iterations=bench_cfg["benchmark_iterations"],
            )
            kv_est = estimate_kv_cache_mb(model, actual_len + config.max_new_tokens)
            result = aggregate_benchmark(
                model_info,
                config,
                raw,
                experiment=f"context_length_{actual_len}",
                context_length=actual_len,
                gpu_util=gpu_util,
                hardware=hardware,
                kv_cache_estimate_mb=kv_est,
                metadata={"model_variant": model_cfg["variant"]},
            )
            all_results.append(result)
            print(
                f"  kv_est={kv_est:.1f}MB peak_vram={result.peak_vram_mb:.0f}MB "
                f"prefill={result.prefill_latency_ms.mean:.1f}ms "
                f"decode={result.decode_tokens_per_second.mean:.1f} tok/s"
            )
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  SKIPPED ctx={ctx_len}: OOM — {exc}")
            torch.cuda.empty_cache()

    out_dir = BENCHMARKS_DIR / "kv_cache"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = BenchmarkRunSummary(config_snapshot=cfg, results=all_results)
    out_path = summary.save(out_dir / f"kv_cache_{ts}.json")
    summary.save(out_dir / "latest_kv_cache.json")
    print(f"\nKV-cache results saved to {out_path}")


if __name__ == "__main__":
    main()
