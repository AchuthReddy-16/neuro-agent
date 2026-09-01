"""PyTorch profiler integration for inference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.engine import generate_with_timings, set_seed
from neuro_agent.paths import RESULTS_DIR


@dataclass
class ProfilerSummary:
    """Summary of hottest operations from a PyTorch profiler run."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    phase: str = ""
    top_cpu_ops: list[dict[str, Any]] = field(default_factory=list)
    top_cuda_ops: list[dict[str, Any]] = field(default_factory=list)
    trace_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "phase": self.phase,
            "top_cpu_ops": self.top_cpu_ops,
            "top_cuda_ops": self.top_cuda_ops,
            "trace_path": self.trace_path,
            "metadata": self.metadata,
        }


def _event_time_us(ka: Any, device: str) -> float:
    """Extract timing from profiler event (API varies across PyTorch versions)."""
    if device == "cuda":
        for attr in ("cuda_time_total", "device_time_total", "cuda_time"):
            if hasattr(ka, attr):
                return float(getattr(ka, attr))
        return 0.0
    return float(ka.cpu_time_total)


def _extract_top_ops(prof: profile, device: str, top_k: int = 15) -> list[dict[str, Any]]:
    key_averages = prof.key_averages()
    if device == "cpu":
        sorted_ka = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:top_k]
    else:
        sorted_ka = sorted(key_averages, key=lambda x: _event_time_us(x, "cuda"), reverse=True)[:top_k]
    return [
        {
            "name": ka.key,
            "cpu_time_us": ka.cpu_time_total,
            "cuda_time_us": _event_time_us(ka, "cuda"),
            "count": ka.count,
        }
        for ka in sorted_ka
    ]


@torch.inference_mode()
def profile_prefill(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: InferenceConfig,
    output_dir: Path,
) -> ProfilerSummary:
    """Profile a single prefill forward pass."""
    device = next(model.parameters()).device
    set_seed(config.seed)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "prefill_trace.json"

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("prefill_forward"):
            model(input_ids=input_ids, use_cache=config.use_cache)
        torch.cuda.synchronize(device)

    prof.export_chrome_trace(str(trace_path))
    summary = ProfilerSummary(
        phase="prefill",
        top_cpu_ops=_extract_top_ops(prof, "cpu"),
        top_cuda_ops=_extract_top_ops(prof, "cuda"),
        trace_path=str(trace_path),
        metadata={"prompt_tokens": input_ids.shape[1], "use_cache": config.use_cache},
    )
    return summary


@torch.inference_mode()
def profile_decode_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: InferenceConfig,
    output_dir: Path,
) -> ProfilerSummary:
    """Profile one decode step after prefill."""
    device = next(model.parameters()).device
    set_seed(config.seed)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    outputs = model(input_ids=input_ids, use_cache=config.use_cache)
    past = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "decode_trace.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("decode_step"):
            if config.use_cache and past is not None:
                model(input_ids=next_token, past_key_values=past, use_cache=True)
            else:
                full = torch.cat([input_ids, next_token], dim=1)
                model(input_ids=full, use_cache=False)
        torch.cuda.synchronize(device)

    prof.export_chrome_trace(str(trace_path))
    return ProfilerSummary(
        phase="decode",
        top_cpu_ops=_extract_top_ops(prof, "cpu"),
        top_cuda_ops=_extract_top_ops(prof, "cuda"),
        trace_path=str(trace_path),
        metadata={"use_cache": config.use_cache},
    )


def run_pytorch_profiling(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: InferenceConfig,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one representative prefill + decode profiling sample."""
    out_dir = output_dir or (RESULTS_DIR / "profiling" / "pytorch")
    prefill_summary = profile_prefill(model, tokenizer, prompt, config, out_dir)
    decode_summary = profile_decode_step(model, tokenizer, prompt, config, out_dir)

    prefill_path = prefill_summary.save(out_dir / "prefill_summary.json")
    decode_path = decode_summary.save(out_dir / "decode_summary.json")

    combined = {
        "prefill": prefill_summary.to_dict(),
        "decode": decode_summary.to_dict(),
    }
    combined_path = out_dir / "profiling_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2))

    return {
        "prefill_summary": str(prefill_path),
        "decode_summary": str(decode_path),
        "combined": str(combined_path),
        "top_cuda_prefill": prefill_summary.top_cuda_ops[:5],
        "top_cuda_decode": decode_summary.top_cuda_ops[:5],
    }
