"""Benchmark execution and GPU sampling utilities."""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from neuro_agent.evaluation.schema import BenchmarkResult, LatencyStats
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.engine import (
    estimate_kv_cache_mb,
    generate_with_timings,
    make_prompt_of_token_length,
)
from neuro_agent.inference.model_loader import ModelLoadInfo


class GpuUtilSampler:
    """Sample GPU utilization via nvidia-smi during a benchmark run."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                val = float(result.stdout.strip().split("\n")[0])
                self.samples.append(val)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> LatencyStats | None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return None
        return LatencyStats.from_values(self.samples)


def run_benchmark_iterations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: InferenceConfig,
    prompt: str,
    warmup_iterations: int,
    benchmark_iterations: int,
    sample_gpu_util: bool = True,
) -> tuple[list[Any], LatencyStats | None]:
    """Run warmup + timed iterations, return raw timing objects."""
    device = next(model.parameters()).device

    for _ in range(warmup_iterations):
        generate_with_timings(model, tokenizer, prompt, config)
        torch.cuda.synchronize(device)

    results = []
    gpu_sampler = GpuUtilSampler() if sample_gpu_util else None
    if gpu_sampler:
        gpu_sampler.start()

    for _ in range(benchmark_iterations):
        out = generate_with_timings(model, tokenizer, prompt, config)
        results.append(out)
        torch.cuda.synchronize(device)

    gpu_stats = gpu_sampler.stop() if gpu_sampler else None
    return results, gpu_stats


def aggregate_benchmark(
    model_info: ModelLoadInfo,
    config: InferenceConfig,
    results: list[Any],
    experiment: str,
    context_length: int,
    gpu_util: LatencyStats | None,
    hardware: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    kv_cache_estimate_mb: float | None = None,
) -> BenchmarkResult:
    """Aggregate per-iteration timings into a BenchmarkResult."""
    prefill = LatencyStats.from_values([r.timings.prefill_latency_ms for r in results])
    ttft = LatencyStats.from_values([r.timings.ttft_ms for r in results])
    decode_per_tok = LatencyStats.from_values(
        [r.timings.decode_latency_per_token_ms for r in results]
    )
    decode_tps = LatencyStats.from_values([r.timings.decode_tokens_per_second for r in results])
    e2e = LatencyStats.from_values([r.timings.end_to_end_latency_ms for r in results])
    peak_vram = LatencyStats.from_values([r.timings.peak_vram_mb for r in results])
    kv_measured = LatencyStats.from_values(
        [r.timings.kv_cache_memory_mb or 0.0 for r in results]
    )

    return BenchmarkResult(
        model_name=model_info.model_name,
        model_variant=metadata.get("model_variant", "base_BF16") if metadata else "base_BF16",
        precision=config.dtype,
        quantization=(
            metadata.get("quantization", getattr(config, "quantization", "none"))
            if metadata
            else getattr(config, "quantization", "none")
        ),
        context_length=context_length,
        generated_tokens=results[-1].timings.generated_token_count if results else 0,
        use_cache=config.use_cache,
        load_time_s=model_info.load_time_s,
        weight_memory_mb=model_info.weight_memory_mb,
        peak_vram_mb=peak_vram.mean,
        kv_cache_memory_mb=kv_cache_estimate_mb or kv_measured.mean,
        prompt_token_count=results[-1].timings.prompt_token_count if results else context_length,
        prefill_latency_ms=prefill,
        ttft_ms=ttft,
        decode_latency_per_token_ms=decode_per_tok,
        decode_tokens_per_second=decode_tps,
        end_to_end_latency_ms=e2e,
        gpu_utilization_pct=gpu_util,
        hardware=hardware,
        experiment=experiment,
        metadata={
            **(metadata or {}),
            "peak_vram_stats": peak_vram,
            "kv_cache_measured_mb": kv_measured.mean,
            "iterations": len(results),
        },
    )


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    base_text: str,
    token_length: int,
) -> tuple[str, int]:
    return make_prompt_of_token_length(tokenizer, base_text, token_length)
