#!/usr/bin/env python3
"""Quick RLVR dataset inspection."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.paths import PROJECT_ROOT


def main() -> None:
    path = PROJECT_ROOT / "data/processed/rlvr_train.jsonl"
    examples = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Total examples: {len(examples)}")
    print(f"Keys: {sorted(examples[0].keys())}")

    task_types = Counter(e.get("task_type", "MISSING") for e in examples)
    print("\nTask type distribution:")
    for k, v in sorted(task_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100 * v / len(examples):.1f}%)")

    vtypes = Counter(e.get("verification_type", "MISSING") for e in examples)
    print("\nVerification type distribution:")
    for k, v in sorted(vtypes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100 * v / len(examples):.1f}%)")

    cross: dict[str, Counter] = defaultdict(Counter)
    for e in examples:
        cross[e.get("task_type")][e.get("verification_type")] += 1
    print("\nTask x Verification cross-tab:")
    for task in sorted(cross):
        print(f"  {task}: {dict(cross[task])}")

    has_tol = sum(1 for e in examples if e.get("tolerance"))
    print(f"\nExamples with tolerance: {has_tol}")

    subjects = Counter()
    for e in examples:
        for s in e.get("source_samples", []):
            subjects[s[:4]] += 1
    print(f"\nUnique subject prefixes: {len(subjects)}")
    print(f"Top subjects: {subjects.most_common(5)}")


if __name__ == "__main__":
    main()
