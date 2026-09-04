"""Model and tokenizer loading utilities."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from neuro_agent.inference.config import InferenceConfig
from neuro_agent.paths import PROJECT_ROOT
from neuro_agent.quantization import (
    apply_int8_runtime,
    build_bitsandbytes_config,
    normalize_quantization,
    QuantizationMethod,
)


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
    quantization: str = "none"
    peak_allocated_mb: float | None = None
    peak_reserved_mb: float | None = None
    allocated_after_load_mb: float | None = None
    nvidia_smi_mb: float | None = None


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
    """Sum parameter storage; falls back for bitsandbytes quantized tensors."""
    total_bytes = 0
    for p in model.parameters():
        if hasattr(p, "quant_state") or p.__class__.__name__ in {"Params4bit", "Int8Params"}:
            # Prefer underlying quantized storage when present
            data = getattr(p, "data", p)
            total_bytes += data.numel() * data.element_size()
            continue
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 * 1024)


def query_nvidia_smi_mb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        values = [float(x.strip()) for x in result.stdout.strip().splitlines() if x.strip()]
        return max(values) if values else None
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None


def load_model_and_tokenizer(
    config: InferenceConfig,
    device: str | torch.device = "cuda:0",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, ModelLoadInfo]:
    """Load HuggingFace causal LM and tokenizer in the requested precision.

    Quantization paths (prefer bitsandbytes / Transformers):
      - none / bf16: full BF16 weights
      - int8: BitsAndBytesConfig(load_in_8bit=True)
      - int4: BitsAndBytes NF4 + double quant (matches QLoRA training)
    LoRA adapters are applied via PEFT after the base load.
    """
    dtype = resolve_dtype(config.dtype)
    device = torch.device(device)
    quant = normalize_quantization(config.quantization)
    bnb_config = build_bitsandbytes_config(quant)

    if device.type == "cuda":
        _prepare_cuda_device(device)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    load_kwargs: dict = {
        "trust_remote_code": config.trust_remote_code,
        "device_map": {"": device},
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        # Original INT8 load omits torch_dtype (H.1B/H.3). Passing bfloat16 with
        # load_in_8bit forces extra per-linear casts and collapses decode throughput.
        if quant == QuantizationMethod.INT8 and config.int8_compute_dtype in {
            "float16",
            "fp16",
        }:
            load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)

    if config.adapter_path:
        adapter_path = Path(config.adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    if quant == QuantizationMethod.INT8 and (
        config.int8_compute_dtype or config.compile_surrounding
    ):
        apply_int8_runtime(
            model,
            compute_dtype=config.int8_compute_dtype,
            compile_surrounding=config.compile_surrounding,
        )
    load_time_s = time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        dev_idx = device.index if device.index is not None else 0
        allocated = torch.cuda.memory_allocated(dev_idx) / (1024 * 1024)
        peak_allocated = torch.cuda.max_memory_allocated(dev_idx) / (1024 * 1024)
        peak_reserved = torch.cuda.max_memory_reserved(dev_idx) / (1024 * 1024)
    else:
        allocated = peak_allocated = peak_reserved = None

    weight_memory_mb = compute_weight_memory_mb(model)
    # For quantized loads, parameter nbytes can under/over-count; prefer CUDA allocated
    if allocated is not None and bnb_config is not None:
        weight_memory_mb = allocated

    num_parameters = sum(p.numel() for p in model.parameters())

    info = ModelLoadInfo(
        model_name=config.model_name,
        load_time_s=load_time_s,
        weight_memory_mb=weight_memory_mb,
        dtype=config.dtype,
        device=str(device),
        num_parameters=num_parameters,
        quantization=quant.value,
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=peak_reserved,
        allocated_after_load_mb=allocated,
        nvidia_smi_mb=query_nvidia_smi_mb(),
    )
    return model, tokenizer, info
