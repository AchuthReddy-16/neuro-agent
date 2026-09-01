"""Post-training quantization (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class QuantizationMethod(str, Enum):
    """Supported quantization precisions for systems evaluation."""

    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class QuantizationConfig:
    """PTQ configuration."""

    method: QuantizationMethod = QuantizationMethod.INT4
    calibration_samples: int = 128
    output_dir: Path = Path("/workspace/neuro-agent/checkpoints/quantized")


class Quantizer:
    """Placeholder post-training quantizer."""

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config

    def quantize(self, model_path: Path) -> Path:
        """Apply post-training quantization. Not implemented in scaffold."""
        raise NotImplementedError("PTQ will be implemented in the quantization stage.")
