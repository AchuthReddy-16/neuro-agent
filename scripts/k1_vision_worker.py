#!/usr/bin/env python3
"""Isolated VLM worker for swap profiling — exits to fully release CUDA."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def gpu_stats() -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, total, util = [float(x.strip()) for x in out.split(",")]
    return {
        "nvidia_smi_used_mb": used,
        "nvidia_smi_total_mb": total,
        "gpu_util_pct": util,
        "torch_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 1),
        "torch_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 1),
    }


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    image_path = Path(payload["image_path"])
    question = payload["question"]
    context = payload.get("context") or {}

    from neuro_agent.inference.config import InferenceConfig
    from neuro_agent.multimodal.dataset import build_multimodal_messages
    from neuro_agent.multimodal.model import load_vlm_for_inference
    from qwen_vl_utils import process_vision_info

    result: dict = {"ok": False}
    before_all = gpu_stats()

    cfg = InferenceConfig(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        dtype="bfloat16",
        trust_remote_code=True,
        adapter_path=str(PROJECT_ROOT / "checkpoints/multimodal_sft_corrected/final"),
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        use_cache=True,
        seed=42,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    before_load = gpu_stats()
    model, processor, info = load_vlm_for_inference(cfg)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    after_load = gpu_stats()

    system_prompt = (
        "You are a neuroscience research assistant analyzing EEG-derived plots. "
        "Answer briefly based on the image and context."
    )
    user_text = (
        f"Context:\n{json.dumps(context, indent=2, sort_keys=True)}\n\n"
        f"Question: {question.strip()}"
    )
    messages = build_multimodal_messages(
        system_prompt=system_prompt,
        user_text=user_text,
        image_uri=f"file://{image_path.resolve()}",
    )

    torch.cuda.reset_peak_memory_stats()
    t_pre0 = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    torch.cuda.synchronize()
    preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

    t_gen0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t_gen0) * 1000.0
    in_len = int(inputs["input_ids"].shape[-1])
    n_new = int(out[0][in_len:].numel())
    decoded = processor.batch_decode(
        out[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    prefill_proxy = gen_ms * (in_len / max(in_len + n_new, 1))

    t_un0 = time.perf_counter()
    before_un = gpu_stats()
    del model
    del processor
    del inputs
    del out
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # brief settle
    time.sleep(1.0)
    after_un = gpu_stats()
    unload_ms = (time.perf_counter() - t_un0) * 1000.0

    result = {
        "ok": True,
        "before_all": before_all,
        "vlm_load_ms": round(load_ms, 3),
        "load_time_s_reported": info.load_time_s,
        "before_load": before_load,
        "after_load": after_load,
        "vram_delta_load_mb": round(
            after_load["nvidia_smi_used_mb"] - before_load["nvidia_smi_used_mb"], 1
        ),
        "preprocess_ms": round(preprocess_ms, 3),
        "generate_ms": round(gen_ms, 3),
        "ttft_proxy_ms": round(preprocess_ms + prefill_proxy, 3),
        "infer_e2e_ms": round(preprocess_ms + gen_ms, 3),
        "input_tokens": in_len,
        "completion_tokens": n_new,
        "output": decoded.strip(),
        "peak_torch_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "vlm_unload_ms": round(unload_ms, 3),
        "before_unload": before_un,
        "after_unload": after_un,
        "vram_released_unload_mb": round(
            before_un["nvidia_smi_used_mb"] - after_un["nvidia_smi_used_mb"], 1
        ),
    }
    out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
