"""Runtime helpers for the bitsandbytes INT8 path.

These keep Linear8bitLt / MatMul8bitLt as the compute backend. They only
change activation dtype and surrounding (non-quantized) execution.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

QUANTIZED_MODULE_NAMES = {"Linear8bitLt", "Linear4bit"}
QUANTIZED_PARAM_NAMES = {"Int8Params", "Params4bit"}
SURROUNDING_MODULE_NAMES = {
    "Qwen3RMSNorm",
    "Qwen2RMSNorm",
    "RMSNorm",
    "Qwen3RotaryEmbedding",
    "Qwen2RotaryEmbedding",
    "RotaryEmbedding",
}


def is_quantized_param(param: torch.Tensor) -> bool:
    return param.__class__.__name__ in QUANTIZED_PARAM_NAMES or hasattr(param, "quant_state")


def convert_non_quantized_to_dtype(model: nn.Module, dtype: torch.dtype) -> dict[str, int]:
    """Cast embeddings, norms, LoRA, and other non-INT8 tensors to `dtype`.

    bitsandbytes MatMul8bitLt always quantizes activations from FP16. If the
    rest of the model stays in BF16, every Linear8bitLt issues a BF16→FP16 copy.
    Running non-quantized compute in FP16 lets INT8 linears receive FP16
    activations without a per-layer cast. Quantized Int8Params are left untouched.
    """
    converted_params = 0
    converted_buffers = 0
    skipped_quantized = 0

    for module in model.modules():
        if type(module).__name__ in QUANTIZED_MODULE_NAMES:
            bias = getattr(module, "bias", None)
            if bias is not None and bias.is_floating_point() and bias.dtype != dtype:
                module.bias.data = module.bias.data.to(dtype)
                converted_params += 1
            continue

        for _, param in module.named_parameters(recurse=False):
            if is_quantized_param(param):
                skipped_quantized += 1
                continue
            if param.is_floating_point() and param.dtype != dtype:
                param.data = param.data.to(dtype)
                converted_params += 1

        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.is_floating_point() and buf.dtype != dtype:
                setattr(module, name, buf.to(dtype))
                converted_buffers += 1

    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "torch_dtype"):
        cfg.torch_dtype = dtype

    return {
        "converted_params": converted_params,
        "converted_buffers": converted_buffers,
        "skipped_quantized": skipped_quantized,
    }


def compile_surrounding_modules(model: nn.Module, *, mode: str = "default") -> dict[str, Any]:
    """torch.compile only RMSNorm / rotary modules (never Linear8bitLt)."""
    compiled_names: list[str] = []
    errors: list[str] = []

    named = dict(model.named_modules())
    for qual_name, module in list(named.items()):
        if type(module).__name__ not in SURROUNDING_MODULE_NAMES:
            continue
        if not qual_name:
            continue
        parent_name, _, child = qual_name.rpartition(".")
        parent = named.get(parent_name, model) if parent_name else model
        try:
            compiled = torch.compile(module, fullgraph=False, mode=mode, dynamic=False)
            setattr(parent, child, compiled)
            compiled_names.append(qual_name)
        except Exception as exc:  # noqa: BLE001 — experiment must record failure
            errors.append(f"{qual_name}: {type(exc).__name__}: {exc}")

    return {
        "compiled_modules": compiled_names,
        "compiled_count": len(compiled_names),
        "errors": errors,
        "mode": mode,
    }


def apply_int8_runtime(
    model: nn.Module,
    *,
    compute_dtype: str | None = None,
    compile_surrounding: bool = False,
    compile_mode: str = "default",
) -> dict[str, Any]:
    """Apply independently selectable INT8 runtime tweaks. Does not replace bitsandbytes."""
    report: dict[str, Any] = {
        "compute_dtype": compute_dtype,
        "dtype_conversion": None,
        "compile": None,
    }
    if compute_dtype in {"float16", "fp16"}:
        report["dtype_conversion"] = convert_non_quantized_to_dtype(model, torch.float16)
    elif compute_dtype in {"bfloat16", "bf16"}:
        report["dtype_conversion"] = convert_non_quantized_to_dtype(model, torch.bfloat16)

    if compile_surrounding:
        report["compile"] = compile_surrounding_modules(model, mode=compile_mode)
    return report
