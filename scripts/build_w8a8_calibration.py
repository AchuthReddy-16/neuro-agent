#!/usr/bin/env python3
"""H.7C — neuroscience-task calibration set (no held-out S026–S030 leakage)."""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neuro_agent.paths import PROJECT_ROOT, configure_hf_cache
from neuro_agent.training.dataset import extract_subjects, load_sft_examples

SOURCE = PROJECT_ROOT / "data/processed/sft_corrective_v2_mixed.jsonl"
TOKENIZER_SRC = PROJECT_ROOT / "checkpoints/text_merged_corrected_bf16"
OUT_DIR = PROJECT_ROOT / "results/quantization/w8a8_int8"
CALIB_JSONL = OUT_DIR / "calibration_prompts.jsonl"
MANIFEST = OUT_DIR / "calibration_manifest.json"

HELD_OUT = {f"S{i:03d}" for i in range(26, 31)}
TARGET_N = 512
FAMILIES = [
    "execution_vs_imagery",
    "movement_task_classification",
    "channel_ranking",
    "band_power_analysis",
    "numerical_reasoning",
    "statistical_comparison",
    "tool_selection",
    "factual_grounding",
]
SYSTEM = (
    "You are a neuroscience research assistant. Answer using only the provided "
    "context. Respond with a concise direct answer only. Do not add unsupported "
    "interpretation."
)


def main() -> None:
    configure_hf_cache()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_sft_examples(SOURCE)
    eligible = []
    dropped_heldout = 0
    for ex in examples:
        subjects = set(extract_subjects(ex))
        if subjects & HELD_OUT:
            dropped_heldout += 1
            continue
        eligible.append(ex)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for ex in eligible:
        by_cat[ex.get("category", "unknown")].append(ex)

    rng = random.Random(42)
    for rows in by_cat.values():
        rng.shuffle(rows)

    # Stratified round-robin so all 8 production families are represented.
    selected: list[dict] = []
    pointers = {c: 0 for c in by_cat}
    cats = [c for c in FAMILIES if by_cat.get(c)] + [
        c for c in by_cat if c not in FAMILIES
    ]
    while len(selected) < TARGET_N:
        progressed = False
        for cat in cats:
            idx = pointers[cat]
            if idx < len(by_cat[cat]) and len(selected) < TARGET_N:
                selected.append(by_cat[cat][idx])
                pointers[cat] = idx + 1
                progressed = True
        if not progressed:
            break

    tok_path = TOKENIZER_SRC if (TOKENIZER_SRC / "tokenizer.json").exists() else "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=False)

    records = []
    token_lens = []
    for ex in selected:
        user_content = (
            f"Context:\n{json.dumps(ex['tool_context'], indent=2, sort_keys=True)}\n\n"
            f"Question: {ex['question'].strip()}"
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_lens.append(len(ids))
        records.append(
            {
                "id": ex.get("id"),
                "category": ex.get("category"),
                "source_samples": ex.get("source_samples"),
                "text": text,
                "n_tokens": len(ids),
            }
        )

    with CALIB_JSONL.open("w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    subjects = sorted({s for rec in records for s in extract_subjects(rec)})
    manifest = {
        "sample_count": len(records),
        "target_count": TARGET_N,
        "source_file": str(SOURCE),
        "source_split": "sft_corrective_v2_mixed train/val-domain examples (adapter training distribution)",
        "held_out_subjects_excluded": sorted(HELD_OUT),
        "dropped_heldout_rows": dropped_heldout,
        "subjects_present": subjects,
        "subject_min": min(subjects) if subjects else None,
        "subject_max": max(subjects) if subjects else None,
        "category_counts": dict(Counter(r["category"] for r in records)),
        "prompt_length_tokens": {
            "min": min(token_lens),
            "max": max(token_lens),
            "mean": round(statistics.mean(token_lens), 1),
            "median": statistics.median(token_lens),
            "p90": sorted(token_lens)[int(0.9 * (len(token_lens) - 1))],
        },
        "why_production_representative": (
            "Calibration uses the same corrected mixed SFT prompts the production "
            "adapter was trained on (execution/imagery, movement, ranking, band, "
            "numeric, statistical, tool selection, factual grounding), formatted "
            "with the production system prompt and chat template. Held-out "
            "evaluation subjects S026–S030 are excluded."
        ),
        "tokenizer": str(tok_path),
        "calibration_prompts_path": str(CALIB_JSONL),
        "generic_datasets_used": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
