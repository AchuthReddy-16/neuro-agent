"""Inference configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Runtime inference configuration."""

    model_name: str
    dtype: str = "bfloat16"
    seed: int = 42
    do_sample: bool = False
    max_new_tokens: int = 64
    use_cache: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False
    adapter_path: str | None = None
    # none|bf16 → full BF16; int8|int4 → bitsandbytes weight-only PTQ at load
    quantization: str = "none"
    # INT8 only: optional "float16" casts non-quantized tensors so Linear8bitLt
    # skips per-layer BF16→FP16 copies. H.4 measured this and rejected it for
    # decode throughput (casts→0 but decode slower). Default None = H.1B path.
    int8_compute_dtype: str | None = None
    # INT8 only: torch.compile RMSNorm / rotary (never Linear8bitLt). H.4 rejected.
    compile_surrounding: bool = False
