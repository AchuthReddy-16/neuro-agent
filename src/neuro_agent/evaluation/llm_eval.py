"""Reusable LLM evaluation harness for neuroscience tasks."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from neuro_agent.evaluation.verifiers import (
    VerificationResult,
    format_eval_prompt,
    verify_example,
)
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.engine import set_seed
from neuro_agent.inference.model_loader import load_model_and_tokenizer


SUBJECT_RE = re.compile(r"^(S\d{3})")


@dataclass
class EvalExampleRecord:
    """One evaluated example with model output and verification."""

    id: str
    category: str
    verification_type: str
    question: str
    ground_truth: Any
    response: str
    generated_tokens: int
    latency_ms: float
    verification: VerificationResult
    subjects: list[str] = field(default_factory=list)


@dataclass
class EvalRunSummary:
    """Aggregated evaluation metrics."""

    model_name: str
    variant: str
    dtype: str
    total_examples: int
    verifier_pass_rate: float
    task_accuracy: float
    invalid_parse_rate: float
    empty_refusal_rate: float
    avg_generated_tokens: float
    runtime_s: float
    examples_per_second: float
    avg_latency_ms: float
    peak_torch_allocated_mb: float
    peak_torch_reserved_mb: float
    nvidia_smi_peak_mb: float | None
    held_out_subjects: list[str]
    task_distribution: dict[str, int]
    verification_type_distribution: dict[str, int]
    unsupported_claim_available: bool
    unsupported_claim_rate: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_eval_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def extract_subjects(example: dict[str, Any]) -> list[str]:
    subjects: list[str] = []
    for sample in example.get("source_samples", []):
        m = SUBJECT_RE.match(str(sample))
        if m:
            subjects.append(m.group(1))
    return subjects


def verify_heldout_integrity(
    examples: list[dict[str, Any]],
    held_out_subjects: set[str],
    forbidden_subjects: set[str],
) -> dict[str, Any]:
    """Fail loudly if held-out set is contaminated by train/validation subjects."""
    all_subjects: set[str] = set()
    violations: list[dict[str, str]] = []

    for ex in examples:
        subjects = extract_subjects(ex)
        all_subjects.update(subjects)
        for subject in subjects:
            if subject in forbidden_subjects:
                violations.append(
                    {
                        "id": ex.get("id", ""),
                        "subject": subject,
                        "reason": "train_or_validation_subject_in_eval",
                    }
                )
            if subject not in held_out_subjects:
                violations.append(
                    {
                        "id": ex.get("id", ""),
                        "subject": subject,
                        "reason": "subject_not_in_held_out_allowlist",
                    }
                )

    if violations:
        sample = violations[:10]
        raise ValueError(
            f"Held-out integrity check failed with {len(violations)} violations. "
            f"Sample: {sample}"
        )

    return {
        "confirmed_subjects": sorted(all_subjects),
        "example_count": len(examples),
        "integrity_passed": True,
    }


def _build_prompt(
    example: dict[str, Any],
    system_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    user_content = (
        f"Context:\n{json.dumps(example.get('context', {}), indent=2, sort_keys=True)}\n\n"
        f"Question: {example['question'].strip()}"
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return format_eval_prompt(example, system_prompt)


def _model_device(model: PreTrainedModel) -> torch.device:
    """Resolve device for BF16 and bitsandbytes/device_map models."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: InferenceConfig,
) -> tuple[str, int]:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = _model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "use_cache": config.use_cache,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        gen_kwargs["temperature"] = config.temperature
        gen_kwargs["top_p"] = config.top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0, prompt_len:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return response, int(new_ids.shape[0])


def _query_nvidia_smi_peak_mb() -> float | None:
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


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    skipped_lines = 0
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Interrupted runs can leave a truncated final line; ignore it and resume.
                skipped_lines += 1
                continue
            existing[row["id"]] = row
    if skipped_lines:
        print(f"Skipped {skipped_lines} malformed prediction lines in {path}")
    return existing


def _row_to_record(row: dict[str, Any]) -> EvalExampleRecord:
    verification_data = row["verification"]
    verification = VerificationResult(
        passed=verification_data["passed"],
        verification_type=row["verification_type"],
        parsed_answer=verification_data.get("parsed_answer"),
        expected=row.get("ground_truth"),
        reason=verification_data.get("reason", ""),
        parse_error=verification_data.get("parse_error", False),
        empty_or_refusal=verification_data.get("empty_or_refusal", False),
        grounded_in_context=verification_data.get("grounded_in_context"),
    )
    return EvalExampleRecord(
        id=row["id"],
        category=row["category"],
        verification_type=row["verification_type"],
        question=row["question"],
        ground_truth=row["ground_truth"],
        response=row["response"],
        generated_tokens=row["generated_tokens"],
        latency_ms=row["latency_ms"],
        verification=verification,
        subjects=row.get("subjects", []),
    )


def _record_to_row(record: EvalExampleRecord) -> dict[str, Any]:
    verification = record.verification
    return {
        "id": record.id,
        "category": record.category,
        "verification_type": record.verification_type,
        "question": record.question,
        "ground_truth": record.ground_truth,
        "response": record.response,
        "generated_tokens": record.generated_tokens,
        "latency_ms": record.latency_ms,
        "subjects": record.subjects,
        "verification": {
            "passed": verification.passed,
            "parsed_answer": verification.parsed_answer,
            "reason": verification.reason,
            "parse_error": verification.parse_error,
            "empty_or_refusal": verification.empty_or_refusal,
            "grounded_in_context": verification.grounded_in_context,
        },
    }


def aggregate_metrics(
    records: list[EvalExampleRecord],
    *,
    model_name: str,
    variant: str,
    dtype: str,
    runtime_s: float,
    peak_torch_allocated_mb: float,
    peak_torch_reserved_mb: float,
    nvidia_smi_peak_mb: float | None,
    held_out_subjects: list[str],
) -> tuple[EvalRunSummary, dict[str, Any], dict[str, Any]]:
    total = len(records)
    passed = sum(1 for r in records if r.verification.passed)
    parse_errors = sum(1 for r in records if r.verification.parse_error)
    empty_refusals = sum(1 for r in records if r.verification.empty_or_refusal)
    avg_tokens = _safe_rate(sum(r.generated_tokens for r in records), total)
    avg_latency = _safe_rate(sum(r.latency_ms for r in records), total)

    task_distribution = Counter(r.category for r in records)
    verification_distribution = Counter(r.verification_type for r in records)

    per_task: dict[str, dict[str, Any]] = {}
    for category in sorted(task_distribution):
        subset = [r for r in records if r.category == category]
        per_task[category] = {
            "count": len(subset),
            "verifier_pass_rate": _safe_rate(sum(1 for r in subset if r.verification.passed), len(subset)),
            "invalid_parse_rate": _safe_rate(sum(1 for r in subset if r.verification.parse_error), len(subset)),
            "empty_refusal_rate": _safe_rate(
                sum(1 for r in subset if r.verification.empty_or_refusal),
                len(subset),
            ),
            "avg_generated_tokens": _safe_rate(sum(r.generated_tokens for r in subset), len(subset)),
        }

    per_verifier: dict[str, dict[str, Any]] = {}
    for vtype in sorted(verification_distribution):
        subset = [r for r in records if r.verification_type == vtype]
        grounded = [r for r in subset if r.verification.grounded_in_context is not None]
        grounded_pass = sum(
            1 for r in grounded if r.verification.grounded_in_context and not r.verification.passed
        )
        per_verifier[vtype] = {
            "count": len(subset),
            "verifier_pass_rate": _safe_rate(sum(1 for r in subset if r.verification.passed), len(subset)),
            "invalid_parse_rate": _safe_rate(sum(1 for r in subset if r.verification.parse_error), len(subset)),
            "empty_refusal_rate": _safe_rate(
                sum(1 for r in subset if r.verification.empty_or_refusal),
                len(subset),
            ),
            "avg_generated_tokens": _safe_rate(sum(r.generated_tokens for r in subset), len(subset)),
            "unsupported_claim_rate": (
                _safe_rate(grounded_pass, len(grounded)) if grounded else None
            ),
        }

    grounded_available = any(r.verification.grounded_in_context is not None for r in records)
    unsupported_claim_rate: float | None = None
    if grounded_available:
        grounded_records = [r for r in records if r.verification.grounded_in_context is not None]
        unsupported_claim_rate = _safe_rate(
            sum(
                1
                for r in grounded_records
                if r.verification.grounded_in_context and not r.verification.passed
            ),
            len(grounded_records),
        )

    summary = EvalRunSummary(
        model_name=model_name,
        variant=variant,
        dtype=dtype,
        total_examples=total,
        verifier_pass_rate=_safe_rate(passed, total),
        task_accuracy=_safe_rate(passed, total),
        invalid_parse_rate=_safe_rate(parse_errors, total),
        empty_refusal_rate=_safe_rate(empty_refusals, total),
        avg_generated_tokens=avg_tokens,
        runtime_s=runtime_s,
        examples_per_second=_safe_rate(total, runtime_s),
        avg_latency_ms=avg_latency,
        peak_torch_allocated_mb=peak_torch_allocated_mb,
        peak_torch_reserved_mb=peak_torch_reserved_mb,
        nvidia_smi_peak_mb=nvidia_smi_peak_mb,
        held_out_subjects=held_out_subjects,
        task_distribution=dict(task_distribution),
        verification_type_distribution=dict(verification_distribution),
        unsupported_claim_available=grounded_available,
        unsupported_claim_rate=unsupported_claim_rate,
    )
    return summary, per_task, per_verifier


def run_llm_evaluation(
    examples: list[dict[str, Any]],
    config: InferenceConfig,
    *,
    system_prompt: str,
    model_name: str,
    variant: str,
    output_dir: Path,
    progress_every: int = 25,
) -> EvalRunSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    model, tokenizer, model_info = load_model_and_tokenizer(config)
    device = _model_device(model)
    if device.type == "cuda":
        dev_idx = device.index if device.index is not None else 0
        torch.cuda.reset_peak_memory_stats(dev_idx)

    records: list[EvalExampleRecord] = []
    failures: list[dict[str, Any]] = []
    predictions_path = output_dir / "predictions.jsonl"
    failures_path = output_dir / "failures.jsonl"

    existing_predictions = _load_existing_predictions(predictions_path)
    if existing_predictions:
        print(f"Resuming evaluation: {len(existing_predictions)} predictions already on disk")

    t0 = time.perf_counter()
    pred_mode = "a" if existing_predictions else "w"
    with predictions_path.open(pred_mode) as pred_f:
        for idx, example in enumerate(examples, start=1):
            example_id = example["id"]
            if example_id in existing_predictions:
                record = _row_to_record(existing_predictions[example_id])
                records.append(record)
                if idx % progress_every == 0 or idx == len(examples):
                    elapsed = time.perf_counter() - t0
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{idx}/{len(examples)}] "
                        f"pass_rate={_safe_rate(sum(1 for r in records if r.verification.passed), idx):.3f} "
                        f"elapsed={elapsed:.1f}s rate={rate:.2f} ex/s (resumed)"
                    )
                continue

            prompt = _build_prompt(example, system_prompt, tokenizer)
            start = time.perf_counter()
            response, generated_tokens = _generate_response(model, tokenizer, prompt, config)
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

            pred_row = _record_to_row(record)
            pred_f.write(json.dumps(pred_row) + "\n")

            if idx % progress_every == 0 or idx == len(examples):
                elapsed = time.perf_counter() - t0
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{idx}/{len(examples)}] "
                    f"pass_rate={_safe_rate(sum(1 for r in records if r.verification.passed), idx):.3f} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} ex/s"
                )

    failures = []
    with failures_path.open("w") as fail_f:
        for record in records:
            if not record.verification.passed:
                fail_row = dict(_record_to_row(record))
                fail_row["failure_reason"] = record.verification.reason
                fail_f.write(json.dumps(fail_row) + "\n")
                failures.append(fail_row)

    runtime_s = time.perf_counter() - t0
    peak_allocated = peak_reserved = 0.0
    if device.type == "cuda":
        dev_idx = device.index if device.index is not None else 0
        peak_allocated = torch.cuda.max_memory_allocated(dev_idx) / (1024 * 1024)
        peak_reserved = torch.cuda.max_memory_reserved(dev_idx) / (1024 * 1024)
    nvidia_peak = _query_nvidia_smi_peak_mb()

    held_out_subjects = sorted({s for r in records for s in r.subjects})
    summary, per_task, per_verifier = aggregate_metrics(
        records,
        model_name=model_name,
        variant=variant,
        dtype=config.dtype,
        runtime_s=runtime_s,
        peak_torch_allocated_mb=peak_allocated,
        peak_torch_reserved_mb=peak_reserved,
        nvidia_smi_peak_mb=nvidia_peak,
        held_out_subjects=held_out_subjects,
    )
    summary.metadata = {
        "model_load_time_s": model_info.load_time_s,
        "weight_memory_mb": model_info.weight_memory_mb,
        "num_parameters": model_info.num_parameters,
        "failure_count": len(failures),
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(asdict(summary), f, indent=2)
    with (output_dir / "per_task_metrics.json").open("w") as f:
        json.dump(per_task, f, indent=2)
    with (output_dir / "verifier_summary.json").open("w") as f:
        json.dump(per_verifier, f, indent=2)

    return summary
