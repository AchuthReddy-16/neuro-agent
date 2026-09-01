#!/usr/bin/env python3
"""Run one representative PyTorch profiler sample."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_benchmark_config
from neuro_agent.evaluation.benchmark import build_prompt
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.model_loader import load_model_and_tokenizer
from neuro_agent.paths import RESULTS_DIR, configure_hf_cache, ensure_dirs
from neuro_agent.profiling.pytorch_profiler import run_pytorch_profiling


def main() -> None:
    configure_hf_cache()
    ensure_dirs()
    cfg = load_benchmark_config()

    model_cfg = cfg["model"]
    inf_cfg = cfg["inference"]
    bench_cfg = cfg["benchmark"]
    prompt_cfg = cfg["prompt"]

    config = InferenceConfig(
        model_name=model_cfg["name"],
        dtype=model_cfg["dtype"],
        seed=inf_cfg["seed"],
        do_sample=inf_cfg["do_sample"],
        max_new_tokens=inf_cfg["max_new_tokens"],
        use_cache=inf_cfg["use_cache"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    print(f"Loading {config.model_name} for profiling...")
    model, tokenizer, info = load_model_and_tokenizer(config)
    prompt, actual_len = build_prompt(
        tokenizer, prompt_cfg["base_text"], bench_cfg["prompt_token_length"]
    )
    print(f"Profiling prompt length: {actual_len} tokens")

    out_dir = RESULTS_DIR / "profiling" / "pytorch"
    result = run_pytorch_profiling(model, tokenizer, prompt, config, out_dir)

    print(f"PyTorch profiling complete:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
