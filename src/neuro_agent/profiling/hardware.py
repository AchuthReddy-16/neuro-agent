"""GPU and CUDA hardware verification."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuro_agent.paths import RESULTS_DIR, ensure_dirs


@dataclass
class HardwareReport:
    """Structured hardware verification report."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=lambda: sys.version)
    platform: str = field(default_factory=platform.platform)
    cuda_available: bool = False
    cuda_version: str | None = None
    cudnn_version: str | None = None
    gpu_count: int = 0
    gpus: list[dict[str, Any]] = field(default_factory=list)
    bf16_supported: bool = False
    cuda_allocation_ok: bool = False
    cuda_allocation_error: str | None = None
    driver_version: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        ensure_dirs()
        out = path or (RESULTS_DIR / "hardware_verify.json")
        out.write_text(json.dumps(self.to_dict(), indent=2))
        return out


def _query_nvidia_smi() -> dict[str, Any]:
    """Query nvidia-smi for driver and GPU info."""
    info: dict[str, Any] = {"driver_version": None, "gpus": []}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                info["gpus"].append(
                    {
                        "name": parts[0],
                        "vram_total_mb": int(float(parts[1])),
                        "driver_version": parts[2],
                        "compute_capability": parts[3],
                    }
                )
                if info["driver_version"] is None:
                    info["driver_version"] = parts[2]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        info["error"] = str(exc)
    return info


def verify_hardware() -> HardwareReport:
    """Run full hardware verification suite."""
    report = HardwareReport()
    smi = _query_nvidia_smi()

    if "error" in smi:
        report.errors.append(f"nvidia-smi: {smi['error']}")
    else:
        report.driver_version = smi.get("driver_version")
        report.gpu_count = len(smi.get("gpus", []))
        report.gpus = smi.get("gpus", [])

    try:
        import torch
    except ImportError:
        report.errors.append("PyTorch not installed; CUDA checks skipped")
        return report

    report.cuda_available = torch.cuda.is_available()
    if report.cuda_available:
        report.cuda_version = torch.version.cuda
        if torch.backends.cudnn.is_available():
            report.cudnn_version = str(torch.backends.cudnn.version())

        report.bf16_supported = torch.cuda.is_bf16_supported()

        # Enrich GPU info from PyTorch if nvidia-smi was unavailable
        if report.gpu_count == 0:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                report.gpus.append(
                    {
                        "name": props.name,
                        "vram_total_mb": props.total_memory // (1024 * 1024),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            report.gpu_count = len(report.gpus)

        # Basic CUDA allocation test
        try:
            device = torch.device("cuda:0")
            x = torch.zeros(1024, 1024, dtype=torch.float32, device=device)
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            del x, y
            torch.cuda.empty_cache()
            report.cuda_allocation_ok = True
        except Exception as exc:
            report.cuda_allocation_ok = False
            report.cuda_allocation_error = str(exc)
            report.errors.append(f"CUDA allocation failed: {exc}")
    else:
        report.errors.append("CUDA not available via PyTorch")

    return report


def main() -> None:
    """CLI entry point for hardware verification."""
    report = verify_hardware()
    out_path = report.save()
    print(f"Hardware report saved to {out_path}")
    print(json.dumps(report.to_dict(), indent=2))
    if report.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
