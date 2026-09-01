"""Model and tokenizer loading utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from neuro_agent.inference.config import InferenceConfig


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class ModelLoadInfo:
    """Metadata captured during model load."""

    model_name: str
    load_time_s: float
    weight_memory_mb: float
    dtype: str
    device: str
    num_parameters: int


def _prepare_cuda_device(device: torch.device) -> int:
    """Initialize CUDA context and return device index."""
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    # reset_peak_memory_stats requires an initialized CUDA context
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev_idx)
    return dev_idx


def resolve_dtype(dtype_str: str) -> torch.dtype:
    key = dtype_str.lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return DTYPE_MAP[key]


def compute_weight_memory_mb(model: PreTrainedModel) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)


def load_model_and_tokenizer(
    config: InferenceConfig,
    device: str | torch.device = "cuda:0",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, ModelLoadInfo]:
    """Load HuggingFace causal LM and tokenizer in the requested precision."""
    dtype = resolve_dtype(config.dtype)
    device = torch.device(device)

    if device.type == "cuda":
        _prepare_cuda_device(device)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
        device_map={"": device},
    )
    model.eval()
    load_time_s = time.perf_counter() - t0

    weight_memory_mb = compute_weight_memory_mb(model)
    num_parameters = sum(p.numel() for p in model.parameters())

    info = ModelLoadInfo(
        model_name=config.model_name,
        load_time_s=load_time_s,
        weight_memory_mb=weight_memory_mb,
        dtype=config.dtype,
        device=str(device),
        num_parameters=num_parameters,
    )
    return model, tokenizer, info
