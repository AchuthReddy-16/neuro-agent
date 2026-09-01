"""Common benchmark result schema for all experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Supported model variants for future stages
MODEL_VARIANTS = (
    "base_BF16",
    "SFT_BF16",
    "RLVR_BF16",
    "base_INT8",
    "base_INT4",
    "SFT_INT4",
    "RLVR_INT4",
)


@dataclass
class LatencyStats:
    """Aggregated latency statistics across benchmark iterations."""

    mean: float
    p50: float
    p95: float
    values: list[float] = field(default_factory=list)

    @classmethod
    def from_values(cls, values: list[float]) -> LatencyStats:
        if not values:
            return cls(mean=0.0, p50=0.0, p95=0.0, values=[])
        sorted_v = sorted(values)
        n = len(sorted_v)

        def percentile(p: float) -> float:
            idx = min(int(p / 100.0 * n), n - 1)
            return sorted_v[idx]

        return cls(
            mean=sum(sorted_v) / n,
            p50=percentile(50),
            p95=percentile(95),
            values=sorted_v,
        )


@dataclass
class BenchmarkResult:
    """Unified schema for inference and systems benchmarks."""

    model_name: str
    model_variant: str
    precision: str
    quantization: str
    context_length: int
    generated_tokens: int
    use_cache: bool
    load_time_s: float | None = None
    weight_memory_mb: float | None = None
    peak_vram_mb: float | None = None
    kv_cache_memory_mb: float | None = None
    prompt_token_count: int | None = None
    prefill_latency_ms: LatencyStats | None = None
    ttft_ms: LatencyStats | None = None
    decode_latency_per_token_ms: LatencyStats | None = None
    decode_tokens_per_second: LatencyStats | None = None
    end_to_end_latency_ms: LatencyStats | None = None
    gpu_utilization_pct: LatencyStats | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    experiment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


@dataclass
class BenchmarkRunSummary:
    """Container for a full benchmark session."""

    config_snapshot: dict[str, Any]
    results: list[BenchmarkResult]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": self.timestamp,
            "config_snapshot": self.config_snapshot,
            "results": [r.to_dict() for r in self.results],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path
