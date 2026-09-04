#!/usr/bin/env python3
"""H.8 — balanced train-only W8A8 calibration (eval-contract formatting)."""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neuro_agent.evaluation.llm_eval import _build_prompt
from neuro_agent.paths import PROJECT_ROOT, configure_hf_cache
from neuro_agent.training.dataset import extract_subjects, load_sft_examples

OUT_DIR = PROJECT_ROOT / "results/quantization/w8a8_int8_quality_repair"
CALIB = OUT_DIR / "calibration_prompts.jsonl"
MANIFEST = OUT_DIR / "calibration_manifest.json"
MIXED = PROJECT_ROOT / "data/processed/sft_corrective_v2_mixed.jsonl"
FEATURES = PROJECT_ROOT / "data/processed/features.parquet"
SAMPLES = PROJECT_ROOT / "data/processed/samples.jsonl"
TOKENIZER_SRC = PROJECT_ROOT / "checkpoints/text_merged_corrected_bf16"
HELD_OUT = {f"S{i:03d}" for i in range(26, 31)}
TRAIN_SUBJECTS = {f"S{i:03d}" for i in range(1, 21)}
TARGET_N = 512
SYSTEM = (
    "You are a neuroscience research assistant. Answer using only the provided "
    "context. Respond with a concise direct answer only. Do not add unsupported "
    "interpretation."
)
EVAL_Q = {
    "execution_vs_imagery": "Is this sample baseline, execution, or imagery?",
    "movement_task_classification": "Return the normalized movement label.",
    "band_power_analysis": "Which channel has the highest alpha_mu_power?",
    "channel_ranking": "Which channel has the highest rms?",
    "numerical_reasoning": "Report the variance for AF3.",
    "statistical_comparison": "Which supplied metric has the larger absolute value?",
    "tool_selection": "Select the appropriate analysis tool.",
}


def _subject_ok(ex: dict) -> bool:
    subs = set(extract_subjects(ex))
    return bool(subs) and not (subs & HELD_OUT) and subs <= TRAIN_SUBJECTS


class _Tok:
    chat_template = True

    def __init__(self, inner):
        self._inner = inner

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return self._inner.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


def _record(category: str, question: str, context: dict, source_samples: list, tokenizer) -> dict:
    example = {"question": question, "context": context}
    text = _build_prompt(example, SYSTEM, tokenizer)
    n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return {
        "id": None,
        "category": category,
        "source_samples": source_samples,
        "text": text,
        "n_tokens": n_tokens,
        "question": question,
        "context": context,
    }


def main() -> None:
    configure_hf_cache()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_SRC), trust_remote_code=False)
    tok_wrap = _Tok(tokenizer)

    mixed = [ex for ex in load_sft_examples(MIXED) if _subject_ok(ex)]
    by_cat: dict[str, list] = defaultdict(list)
    for ex in mixed:
        by_cat[ex.get("category", "unknown")].append(ex)
    for rows in by_cat.values():
        rng.shuffle(rows)

    selected: list[dict] = []

    exec_rows = by_cat["execution_vs_imagery"]
    hard = []
    for ex in exec_rows:
        inp = (ex.get("tool_context") or {}).get("inputs") or {}
        gf = ex.get("grounded_facts") or {}
        movement = inp.get("movement") or gf.get("movement")
        label = ex.get("ground_truth") or gf.get("task_type")
        if movement == "rest" and label not in {"baseline", "baseline_rest"}:
            hard.append(ex)
    # All unique exec + upsample hard rest≠baseline cases.
    exec_pool = list(exec_rows) + hard + hard
    rng.shuffle(exec_pool)
    exec_pool = exec_pool[:252]
    for ex in exec_pool:
        inp = (ex.get("tool_context") or {}).get("inputs") or {}
        gf = ex.get("grounded_facts") or {}
        ctx = {
            "condition": inp.get("condition") or gf.get("condition"),
            "movement": inp.get("movement") or gf.get("movement"),
        }
        rec = _record(
            "execution_vs_imagery",
            EVAL_Q["execution_vs_imagery"],
            ctx,
            ex.get("source_samples") or [],
            tok_wrap,
        )
        rec["id"] = ex.get("id")
        rec["upsampled_hard_rest"] = ex in hard
        selected.append(rec)

    def take_mixed(cat: str, n: int, ctx_fn, question: str) -> None:
        rows = list(by_cat.get(cat, []))
        rng.shuffle(rows)
        for ex in rows[:n]:
            rec = _record(cat, question, ctx_fn(ex), ex.get("source_samples") or [], tok_wrap)
            rec["id"] = ex.get("id")
            selected.append(rec)

    def movement_ctx(ex: dict) -> dict:
        inp = (ex.get("tool_context") or {}).get("inputs") or {}
        return {
            "run_id": inp.get("run_id"),
            "event_code": inp.get("event_code"),
            "task_type": inp.get("task_type"),
        }

    take_mixed(
        "movement_task_classification",
        52,
        movement_ctx,
        EVAL_Q["movement_task_classification"],
    )

    feats = pd.read_parquet(FEATURES)
    feats = feats[feats["split"] == "train"]
    feats = feats[~feats["subject_id"].isin(sorted(HELD_OUT))]
    grouped = {sid: g for sid, g in feats.groupby("sample_id", sort=False)}
    sample_ids = list(grouped.keys())
    rng.shuffle(sample_ids)

    def feature_block(sample_id: str, metric: str) -> dict:
        sub = grouped[sample_id]
        values = {str(ch): float(val) for ch, val in zip(sub["channel"], sub[metric])}
        return {"values": values, "metric": metric}

    # band / ranking / numeric from train-split features (eval templates, no held-out).
    sid_i = 0

    def next_sids(k: int) -> list[str]:
        nonlocal sid_i
        out = sample_ids[sid_i : sid_i + k]
        sid_i += k
        return out

    for sid in next_sids(52):
        selected.append(
            _record(
                "band_power_analysis",
                EVAL_Q["band_power_analysis"],
                feature_block(sid, "alpha_mu_power"),
                [sid],
                tok_wrap,
            )
        )
    for sid in next_sids(52):
        selected.append(
            _record(
                "channel_ranking",
                EVAL_Q["channel_ranking"],
                feature_block(sid, "rms"),
                [sid],
                tok_wrap,
            )
        )
    for sid in next_sids(52):
        row = grouped[sid][grouped[sid]["channel"] == "AF3"]
        if row.empty:
            continue
        r = row.iloc[0]
        selected.append(
            _record(
                "numerical_reasoning",
                EVAL_Q["numerical_reasoning"],
                {"channel": "AF3", "variance": float(r["variance"])},
                [sid],
                tok_wrap,
            )
        )

    for sid in next_sids(18):
        row = grouped[sid][grouped[sid]["channel"] == "AF3"]
        if row.empty:
            continue
        r = row.iloc[0]
        selected.append(
            _record(
                "statistical_comparison",
                EVAL_Q["statistical_comparison"],
                {"mean": float(r["mean"]), "rms": float(r["rms"])},
                [sid],
                tok_wrap,
            )
        )

    tool_requests = [
        "compare beta power across channels",
        "map event codes to movement labels",
        "compute mean amplitude",
        "rank channels by rms",
        "inspect mu/alpha band power",
    ]
    tools = ["band_power", "event_mapper", "mean"]
    for i in range(17):
        sid = sample_ids[(sid_i + i) % len(sample_ids)]
        selected.append(
            _record(
                "tool_selection",
                EVAL_Q["tool_selection"],
                {
                    "requested": tool_requests[i % len(tool_requests)],
                    "available_tools": tools,
                },
                [sid],
                tok_wrap,
            )
        )

    # factual from train sample metadata (not held-out ids)
    samples = []
    with SAMPLES.open() as handle:
        for line in handle:
            ex = json.loads(line)
            if ex.get("split") != "train":
                continue
            if str(ex.get("subject_id")) in HELD_OUT:
                continue
            samples.append(ex)
    rng.shuffle(samples)
    for ex in samples[:17]:
        sid = ex["sample_id"]
        q = (
            f"For sample {sid}, combine the supplied task type and movement into "
            "the normalized condition label `<task_type>_<movement>`. Return only that label."
        )
        selected.append(
            _record(
                "factual_grounding",
                q,
                {
                    "sample_id": sid,
                    "run_id": ex.get("run_id"),
                    "event_code": ex.get("event_code"),
                    "task_type": ex.get("task_type"),
                    "movement": ex.get("movement"),
                },
                [sid],
                tok_wrap,
            )
        )

    selected = selected[:TARGET_N]
    lens = [r["n_tokens"] for r in selected]
    with CALIB.open("w") as handle:
        for rec in selected:
            handle.write(json.dumps(rec) + "\n")

    subjects = sorted({s for rec in selected for s in extract_subjects(rec)})
    manifest = {
        "sample_count": len(selected),
        "target_count": TARGET_N,
        "held_out_subjects_excluded": sorted(HELD_OUT),
        "subjects_present": subjects,
        "category_counts": dict(Counter(r["category"] for r in selected)),
        "execution_vs_imagery_share": Counter(r["category"] for r in selected)[
            "execution_vs_imagery"
        ]
        / len(selected),
        "hard_rest_not_baseline_upsampled": sum(
            1 for r in selected if r.get("upsampled_hard_rest")
        ),
        "prompt_length_tokens": {
            "min": min(lens),
            "max": max(lens),
            "mean": round(statistics.mean(lens), 1),
            "median": statistics.median(lens),
            "p90": sorted(lens)[int(0.9 * (len(lens) - 1))],
        },
        "eval_contract_formatting": True,
        "generic_datasets_used": False,
        "sources": [
            "sft_corrective_v2_mixed.jsonl (execution/movement)",
            "features.parquet split=train (band/ranking/numeric/statistical)",
            "samples.jsonl split=train (factual)",
            "constructed tool prompts on train sample ids",
        ],
        "calibration_prompts_path": str(CALIB),
        "vs_h7": {
            "h7_execution_vs_imagery": 103,
            "h8_execution_vs_imagery": Counter(r["category"] for r in selected)[
                "execution_vs_imagery"
            ],
            "h7_missing_families_now_included": [
                "statistical_comparison",
                "tool_selection",
                "factual_grounding",
            ],
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
