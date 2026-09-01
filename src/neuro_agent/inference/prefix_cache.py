"""Prefix-cache experiment interface for future agent workloads.

Future neuroscience agent prompt structure
------------------------------------------

Every request will share a stable prefix:

    [system prompt]
    [tool definitions]
    [research policy]

followed by a changing suffix:

    [user question / task]

Two benchmark scenarios (not run in this stage)
-----------------------------------------------

1. **Cold prefix** — full prompt processed on every request (no reuse).
2. **Reused prefix** — stable prefix KV cache is retained across requests;
   only the suffix is prefilled and decoded.

This module defines the interface for later benchmarking. No experiment
is executed here.

Later benchmarking options
--------------------------

**vLLM** (recommended for prefix caching):
  - Enable automatic prefix caching (APC) or manual prefix caching APIs.
  - Compare TTFT and prefill latency for cold vs warm prefix.
  - Measure KV cache hit rate and VRAM for cached blocks.

**HuggingFace + manual cache**:
  - Store `past_key_values` from prefix prefill and append suffix tokens.
  - Useful for correctness checks but not production serving.

**TensorRT-LLM / SGLang**:
  - Alternative serving backends with radix/prefix attention.

Example future usage::

    scenario = PrefixCacheScenario(
        system_prompt="You are a neuroscience research assistant.",
        tool_definitions=load_tools_yaml(),
        research_policy=load_policy(),
        user_question="Summarize hippocampal replay evidence.",
    )
    cold = scenario.build_cold_prompt(tokenizer)
    warm_prefix, suffix = scenario.build_warm_prefix_and_suffix(tokenizer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedTokenizerBase


@dataclass
class PrefixCacheScenario:
    """Template for cold-vs-reused-prefix benchmarking (future stage)."""

    system_prompt: str
    tool_definitions: str
    research_policy: str
    user_question: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shared_prefix(self) -> str:
        return (
            f"{self.system_prompt.strip()}\n\n"
            f"{self.tool_definitions.strip()}\n\n"
            f"{self.research_policy.strip()}\n\n"
        )

    @property
    def suffix(self) -> str:
        return self.user_question.strip()

    def build_cold_prompt(self) -> str:
        """Full prompt — prefix processed every time."""
        return f"{self.shared_prefix}{self.suffix}"

    def build_warm_prefix_and_suffix(self) -> tuple[str, str]:
        """Split for reused-prefix experiments."""
        return self.shared_prefix, self.suffix

    def tokenize_cold(self, tokenizer: PreTrainedTokenizerBase) -> list[int]:
        return tokenizer.encode(self.build_cold_prompt(), add_special_tokens=True)

    def tokenize_prefix_suffix(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> tuple[list[int], list[int]]:
        prefix_ids = tokenizer.encode(self.shared_prefix, add_special_tokens=True)
        suffix_ids = tokenizer.encode(self.suffix, add_special_tokens=False)
        return prefix_ids, suffix_ids


@dataclass
class PrefixCacheBenchmarkPlan:
    """Planned experiment metadata — not executed in this stage."""

    backend: str = "vllm"  # or "hf_manual", "sglang", "trt-llm"
    scenarios: list[PrefixCacheScenario] = field(default_factory=list)
    metrics: list[str] = field(
        default_factory=lambda: [
            "prefill_latency_ms",
            "ttft_ms",
            "decode_tokens_per_second",
            "peak_vram_mb",
            "prefix_cache_hit",
        ]
    )
    notes: str = (
        "Run after base BF16 benchmarks. Compare cold vs warm prefix TTFT "
        "and prefill cost using the chosen serving backend."
    )
