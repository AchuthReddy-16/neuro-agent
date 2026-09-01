#!/usr/bin/env python3
"""Orchestrate stage benchmarks: hardware verify, baseline, kv-cache, profiling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def run_step(name: str, cmd: list[str]) -> dict:
    print(f"\n{'='*60}\n>>> {name}\n{'='*60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return {"step": name, "returncode": result.returncode, "cmd": cmd}


def main() -> None:
    python = sys.executable
    steps = [
        ("hardware_verify", [python, str(SCRIPTS / "verify_hardware.py")]),
        ("baseline_benchmark", [python, str(SCRIPTS / "run_baseline_benchmark.py")]),
        ("kv_cache_benchmark", [python, str(SCRIPTS / "run_kv_cache_benchmark.py")]),
        ("pytorch_profiler", [python, str(SCRIPTS / "run_pytorch_profiler.py")]),
    ]

    results = []
    for name, cmd in steps:
        results.append(run_step(name, cmd))

    # Check nsys / ncu availability
    nsys_avail = subprocess.run(["which", "nsys"], capture_output=True).returncode == 0
    ncu_avail = subprocess.run(["which", "ncu"], capture_output=True).returncode == 0

    report = {
        "steps": results,
        "nsys_available": nsys_avail,
        "ncu_available": ncu_avail,
    }

    out = PROJECT_ROOT / "results" / "stage_run_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nStage run report: {out}")


if __name__ == "__main__":
    main()
