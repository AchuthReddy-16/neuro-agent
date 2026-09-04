"""Post-training quantization helpers for text-model systems evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch


class QuantizationMethod(str, Enum):
    """Supported quantization precisions for systems evaluation."""

    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    NONE = "none"


@dataclass
class QuantizationConfig:
    """Load-time PTQ configuration (bitsandbytes / Transformers)."""

    method: QuantizationMethod = QuantizationMethod.BF16
    # INT4 defaults match QLoRA training configs in this repo
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    calibration_samples: int = 128
    output_dir: Path = Path("/workspace/neuro-agent/checkpoints/quantized")


def normalize_quantization(value: str | QuantizationMethod | None) -> QuantizationMethod:
    """Map config strings to QuantizationMethod."""
    if value is None:
        return QuantizationMethod.NONE
    if isinstance(value, QuantizationMethod):
        return value
    key = str(value).strip().lower()
    # Handle accidental Enum stringification
    if key.startswith("quantizationmethod."):
        key = key.split(".", 1)[1]
    aliases = {
        "none": QuantizationMethod.NONE,
        "bf16": QuantizationMethod.BF16,
        "bfloat16": QuantizationMethod.BF16,
        "int8": QuantizationMethod.INT8,
        "8bit": QuantizationMethod.INT8,
        "int4": QuantizationMethod.INT4,
        "4bit": QuantizationMethod.INT4,
        "nf4": QuantizationMethod.INT4,
    }
    if key not in aliases:
        raise ValueError(f"Unsupported quantization method: {value}")
    return aliases[key]


def build_bitsandbytes_config(method: QuantizationMethod | str) -> Any | None:
    """Return BitsAndBytesConfig for INT8/INT4, or None for BF16/none."""
    method = normalize_quantization(method)
    if method in (QuantizationMethod.NONE, QuantizationMethod.BF16):
        return None

    from transformers import BitsAndBytesConfig

    if method == QuantizationMethod.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)

    if method == QuantizationMethod.INT4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    raise ValueError(f"No BitsAndBytesConfig for {method}")


def expected_weight_vram_mb(num_params: int, method: QuantizationMethod | str) -> float:
    """Rough weight-only VRAM estimate (MB) for a dense causal LM."""
    method = normalize_quantization(method)
    bytes_per = {
        QuantizationMethod.NONE: 2.0,
        QuantizationMethod.BF16: 2.0,
        QuantizationMethod.INT8: 1.0,
        QuantizationMethod.INT4: 0.5,
    }[method]
    return (num_params * bytes_per) / (1024 * 1024)


def method_limitations(method: QuantizationMethod | str) -> list[str]:
    """Document practical limitations for each load path."""
    method = normalize_quantization(method)
    common = [
        "Weight-only PTQ via bitsandbytes (Transformers); not GPTQ/AWQ/vLLM.",
        "LoRA adapters load via PEFT on top of the quantized (or BF16) base.",
        "Adapter weights remain BF16/FP16; only base linear weights are quantized.",
    ]
    if method in (QuantizationMethod.NONE, QuantizationMethod.BF16):
        return common + [
            "Full BF16 weights; highest quality reference, highest VRAM.",
            "Expected weight VRAM ≈ 2 bytes/param (~8 GB for a 4B model).",
        ]
    if method == QuantizationMethod.INT8:
        return common + [
            "bitsandbytes LLM.int8() (load_in_8bit); mixed-precision matmul.",
            "Expected weight VRAM ≈ 1 byte/param (~4 GB for a 4B model).",
            "May be slower than BF16 on RTX 4090 if INT8 kernels are not faster than Tensor Cores.",
            "Optional H.4 runtime: int8_compute_dtype=float16 removes BF16→FP16 "
            "casts but was rejected (decode regression); default remains None.",
        ]
    return common + [
        "bitsandbytes NF4 + double quant (same recipe as QLoRA training).",
        "Expected weight VRAM ≈ 0.5 bytes/param (~2 GB for a 4B model).",
        "Compute dtype remains BF16; quality may degrade vs BF16 reference.",
    ]


def apply_int8_runtime(*args, **kwargs):
    """Lazy export so importing quantization helpers does not require CUDA graphs."""
    from neuro_agent.quantization.int8_runtime import apply_int8_runtime as _apply

    return _apply(*args, **kwargs)


class Quantizer:
    """Legacy scaffold entrypoint; load-time PTQ is handled by model_loader."""

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config

    def quantize(self, model_path: Path) -> Path:
        """Offline calibration PTQ is not required for bitsandbytes load-time path."""
        raise NotImplementedError(
            "Offline PTQ not used; load with InferenceConfig.quantization=int8|int4."
        )
