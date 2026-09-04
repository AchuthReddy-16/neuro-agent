"""Multimodal vision-language evaluation harness."""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info

from neuro_agent.evaluation.llm_eval import (
    EvalExampleRecord,
    EvalRunSummary,
    _load_existing_predictions,
    _record_to_row,
    _row_to_record,
    _safe_rate,
    aggregate_metrics,
    extract_subjects,
    verify_heldout_integrity,
)
from neuro_agent.evaluation.verifiers import verify_example
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.engine import set_seed
from neuro_agent.multimodal.dataset import (
    build_multimodal_messages,
    normalize_eval_example,
    resolve_image_path,
)
from neuro_agent.multimodal.model import load_vlm_for_inference
from neuro_agent.paths import PROJECT_ROOT


@dataclass
class MultimodalEvalConfig:
    system_prompt: str
    model_name: str
    variant: str
    output_dir: Path
    project_root: Path = PROJECT_ROOT
    progress_every: int = 25


def _build_eval_messages(
    example: dict[str, Any],
    system_prompt: str,
    project_root: Path,
) -> list[dict[str, Any]]:
    image_path = resolve_image_path(project_root, example["image_path"])
    user_text = (
        f"Context:\n{json.dumps(example.get('context', {}), indent=2, sort_keys=True)}\n\n"
        f"Question: {example['question'].strip()}"
    )
    return build_multimodal_messages(
        system_prompt=system_prompt,
        user_text=user_text,
        image_uri=f"file://{image_path}",
    )


def _generate_multimodal_response(
    model,
    processor,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> tuple[str, int]:
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
    prompt_len = inputs["input_ids"].shape[-1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0, prompt_len:]
    response = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return response, int(new_ids.shape[0])


def _query_nvidia_smi_peak_mb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        values = [float(x.strip()) for x in result.stdout.strip().splitlines() if x.strip()]
        return max(values) if values else None
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None


def run_multimodal_evaluation(
    examples: list[dict[str, Any]],
    inference_config: InferenceConfig,
    eval_config: MultimodalEvalConfig,
) -> EvalRunSummary:
    output_dir = eval_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(inference_config.seed)

    normalized = [normalize_eval_example(ex) for ex in examples]
    model, processor, model_info = load_vlm_for_inference(inference_config)
    if model.device.type == "cuda":
        dev_idx = model.device.index if model.device.index is not None else 0
        torch.cuda.reset_peak_memory_stats(dev_idx)

    records: list[EvalExampleRecord] = []
    predictions_path = output_dir / "predictions.jsonl"
    failures_path = output_dir / "failures.jsonl"
    existing_predictions = _load_existing_predictions(predictions_path)
    if existing_predictions:
        print(f"Resuming evaluation: {len(existing_predictions)} predictions already on disk")

    t0 = time.perf_counter()
    pred_mode = "a" if existing_predictions else "w"
    with predictions_path.open(pred_mode) as pred_f:
        for idx, example in enumerate(normalized, start=1):
            example_id = example["id"]
            if example_id in existing_predictions:
                record = _row_to_record(existing_predictions[example_id])
                records.append(record)
                if idx % eval_config.progress_every == 0 or idx == len(normalized):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[{idx}/{len(normalized)}] "
                        f"pass_rate={_safe_rate(sum(1 for r in records if r.verification.passed), idx):.3f} "
                        f"elapsed={elapsed:.1f}s (resumed)"
                    )
                continue

            messages = _build_eval_messages(
                example,
                eval_config.system_prompt,
                eval_config.project_root,
            )
            start = time.perf_counter()
            response, generated_tokens = _generate_multimodal_response(
                model,
                processor,
                messages,
                max_new_tokens=inference_config.max_new_tokens,
                do_sample=inference_config.do_sample,
                temperature=inference_config.temperature,
                top_p=inference_config.top_p,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            verification = verify_example(example, response)
            record = EvalExampleRecord(
                id=example["id"],
                category=example["category"],
                verification_type=example["verification_type"],
                question=example["question"],
                ground_truth=example["ground_truth"],
                response=response,
                generated_tokens=generated_tokens,
                latency_ms=latency_ms,
                verification=verification,
                subjects=extract_subjects(example),
            )
            records.append(record)
            pred_f.write(json.dumps(_record_to_row(record)) + "\n")

            if idx % eval_config.progress_every == 0 or idx == len(normalized):
                elapsed = time.perf_counter() - t0
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{idx}/{len(normalized)}] "
                    f"pass_rate={_safe_rate(sum(1 for r in records if r.verification.passed), idx):.3f} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} ex/s"
                )

    failures: list[dict[str, Any]] = []
    with failures_path.open("w") as fail_f:
        for record in records:
            if not record.verification.passed:
                fail_row = dict(_record_to_row(record))
                fail_row["failure_reason"] = record.verification.reason
                fail_f.write(json.dumps(fail_row) + "\n")
                failures.append(fail_row)

    runtime_s = time.perf_counter() - t0
    peak_allocated = peak_reserved = 0.0
    if model.device.type == "cuda":
        dev_idx = model.device.index if model.device.index is not None else 0
        peak_allocated = torch.cuda.max_memory_allocated(dev_idx) / (1024 * 1024)
        peak_reserved = torch.cuda.max_memory_reserved(dev_idx) / (1024 * 1024)
    nvidia_peak = _query_nvidia_smi_peak_mb()

    held_out_subjects = sorted({s for r in records for s in r.subjects})
    summary, per_task, per_verifier = aggregate_metrics(
        records,
        model_name=eval_config.model_name,
        variant=eval_config.variant,
        dtype=inference_config.dtype,
        runtime_s=runtime_s,
        peak_torch_allocated_mb=peak_allocated,
        peak_torch_reserved_mb=peak_reserved,
        nvidia_smi_peak_mb=nvidia_peak,
        held_out_subjects=held_out_subjects,
    )
    summary.metadata = {
        "model_load_time_s": model_info.load_time_s,
        "weight_memory_mb": model_info.weight_memory_mb,
        "num_parameters": model_info.total_parameters,
        "failure_count": len(failures),
        "seed": inference_config.seed,
        "max_new_tokens": inference_config.max_new_tokens,
        "do_sample": inference_config.do_sample,
        "multimodal": True,
        "vision_parameters": model_info.vision_parameters,
        "frozen_vision_tower": model_info.frozen_vision_tower,
        "text_checkpoint_reused": model_info.text_checkpoint_reused,
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(asdict(summary), f, indent=2)
    with (output_dir / "per_task_metrics.json").open("w") as f:
        json.dump(per_task, f, indent=2)
    with (output_dir / "verifier_summary.json").open("w") as f:
        json.dump(per_verifier, f, indent=2)

    return summary


__all__ = [
    "MultimodalEvalConfig",
    "run_multimodal_evaluation",
    "verify_heldout_integrity",
]
