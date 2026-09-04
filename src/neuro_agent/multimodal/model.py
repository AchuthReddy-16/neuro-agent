"""Vision-language model loading utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
)

from neuro_agent.inference.config import InferenceConfig
from neuro_agent.paths import PROJECT_ROOT


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


@dataclass
class VLMLoadInfo:
    model_name: str
    load_time_s: float
    total_parameters: int
    vision_parameters: int
    merger_parameters: int
    dtype: str
    device: str
    weight_memory_mb: float
    frozen_vision_tower: bool
    lora_target_modules: list[str]
    text_checkpoint_reused: bool


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    return DTYPE_MAP[dtype_name.lower()]


def _count_params(model: PreTrainedModel, pattern: str) -> int:
    return sum(p.numel() for name, p in model.named_parameters() if pattern in name)


def _freeze_vision_tower(model: PreTrainedModel) -> None:
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False


def print_architecture_summary(info: VLMLoadInfo, qlora_cfg: dict[str, Any]) -> None:
    print("\n=== Multimodal VLM Architecture ===")
    print(f"Selected model:        {info.model_name}")
    print(
        "Why chosen:            Qwen-family VLM, open weights, HF/Transformers support, "
        "fits RTX 4090 24GB with QLoRA"
    )
    print(f"Total parameters:      {info.total_parameters / 1e9:.2f}B")
    print(f"Vision parameters:     {info.vision_parameters / 1e9:.2f}B")
    print(f"Merger parameters:     {info.merger_parameters / 1e6:.1f}M")
    print(f"Precision:             {info.dtype} (4-bit NF4 base + {qlora_cfg.get('bnb_4bit_compute_dtype', 'bf16')} compute)")
    print(f"Expected VRAM (load):  ~{info.weight_memory_mb:.0f} MB weights + activations")
    print(f"Frozen vision tower:   {info.frozen_vision_tower}")
    print(f"LoRA modules:          {', '.join(info.lora_target_modules)}")
    print(
        "Text checkpoint reuse: "
        + ("yes" if info.text_checkpoint_reused else "no — separate VLM LoRA branch (Qwen3 text LoRA incompatible)")
    )
    print("===================================\n")


def load_vlm_for_training(
    model_cfg: dict[str, Any],
    qlora_cfg: dict[str, Any],
    *,
    freeze_vision_tower: bool = True,
) -> tuple[PreTrainedModel, Any, VLMLoadInfo]:
    model_name = model_cfg["name"]
    compute_dtype = _resolve_dtype(qlora_cfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
    )

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = prepare_model_for_kbit_training(model)

    if freeze_vision_tower:
        _freeze_vision_tower(model)

    base_adapter_path = model_cfg.get("adapter_path")
    if base_adapter_path:
        adapter_dir = Path(base_adapter_path)
        if not adapter_dir.is_absolute():
            adapter_dir = PROJECT_ROOT / adapter_dir
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        lora_target_modules = list(qlora_cfg["target_modules"])
    else:
        lora_config = LoraConfig(
            r=int(qlora_cfg["lora_r"]),
            lora_alpha=int(qlora_cfg["lora_alpha"]),
            lora_dropout=float(qlora_cfg["lora_dropout"]),
            target_modules=list(qlora_cfg["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        lora_target_modules = list(qlora_cfg["target_modules"])

    load_time_s = time.perf_counter() - t0
    total_parameters = sum(p.numel() for p in model.parameters())
    vision_parameters = _count_params(model, "visual")
    merger_parameters = _count_params(model, "merger")
    weight_memory_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    info = VLMLoadInfo(
        model_name=model_name,
        load_time_s=load_time_s,
        total_parameters=total_parameters,
        vision_parameters=vision_parameters,
        merger_parameters=merger_parameters,
        dtype="4bit_nf4",
        device="cuda:0",
        weight_memory_mb=weight_memory_mb,
        frozen_vision_tower=freeze_vision_tower,
        lora_target_modules=lora_target_modules,
        text_checkpoint_reused=bool(model_cfg.get("adapter_path")),
    )
    return model, processor, info


def load_vlm_for_inference(
    config: InferenceConfig,
    *,
    model_class: type[PreTrainedModel] = Qwen2_5_VLForConditionalGeneration,
) -> tuple[PreTrainedModel, Any, VLMLoadInfo]:
    dtype = _resolve_dtype(config.dtype)
    device = torch.device("cuda:0")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    model = model_class.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=config.trust_remote_code,
    )
    if config.adapter_path:
        adapter_path = Path(config.adapter_path)
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    load_time_s = time.perf_counter() - t0
    total_parameters = sum(p.numel() for p in model.parameters())
    vision_parameters = _count_params(model, "visual")
    merger_parameters = _count_params(model, "merger")
    weight_memory_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    info = VLMLoadInfo(
        model_name=config.model_name,
        load_time_s=load_time_s,
        total_parameters=total_parameters,
        vision_parameters=vision_parameters,
        merger_parameters=merger_parameters,
        dtype=config.dtype,
        device=str(device),
        weight_memory_mb=weight_memory_mb,
        frozen_vision_tower=True,
        lora_target_modules=[],
        text_checkpoint_reused=bool(config.adapter_path),
    )
    return model, processor, info
