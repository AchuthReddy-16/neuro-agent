"""Deterministic HuggingFace inference with prefill/decode separation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from neuro_agent.inference.config import InferenceConfig


@dataclass
class GenerationTimings:
    """Per-run timing breakdown in milliseconds."""

    prompt_token_count: int
    generated_token_count: int
    prefill_latency_ms: float
    ttft_ms: float
    decode_latency_per_token_ms: float
    decode_tokens_per_second: float
    end_to_end_latency_ms: float
    peak_vram_mb: float
    kv_cache_memory_mb: float | None = None


@dataclass
class GenerationOutput:
    """Full generation result with timings and text."""

    text: str
    token_ids: list[int]
    timings: GenerationTimings


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_prompt_of_token_length(
    tokenizer: PreTrainedTokenizerBase,
    base_text: str,
    target_length: int,
) -> tuple[str, int]:
    """Build a prompt with exactly target_length tokens."""
    seed_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not seed_ids:
        seed_ids = [tokenizer.eos_token_id or 0]

    ids = list(seed_ids)
    while len(ids) < target_length:
        ids.extend(seed_ids)
    ids = ids[:target_length]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    # Re-truncate if decode/encode round-trip changed length
    if actual != target_length:
        ids = ids[:target_length]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text, actual


def estimate_kv_cache_mb(
    model: PreTrainedModel,
    seq_len: int,
    batch_size: int = 1,
) -> float | None:
    """Estimate KV cache memory from model config if available."""
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", None))
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None and hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads"):
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    hidden = getattr(cfg, "hidden_size", None)

    if None in (n_layers, n_kv_heads, head_dim):
        return None

    # K and V per layer: 2 * batch * seq * n_kv_heads * head_dim * bytes
    bytes_per_elem = 2  # bf16
    kv_bytes = 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * bytes_per_elem
    return kv_bytes / (1024 * 1024)


def _next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


@torch.inference_mode()
def generate_with_timings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    config: InferenceConfig,
) -> GenerationOutput:
    """Run greedy/deterministic generation with explicit prefill and decode timing."""
    device = next(model.parameters()).device
    set_seed(config.seed)

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    prompt_token_count = input_ids.shape[1]

    torch.cuda.synchronize(device)
    if device.type == "cuda":
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            torch.cuda.reset_peak_memory_stats(dev_idx)
        except RuntimeError:
            pass  # context may not be ready; peak stats still valid after sync
    mem_before = torch.cuda.memory_allocated(device)

    # --- Prefill ---
    t_start = time.perf_counter()
    t_prefill_start = t_start

    outputs = model(input_ids=input_ids, use_cache=config.use_cache)
    logits = outputs.logits
    past_key_values = outputs.past_key_values

    torch.cuda.synchronize(device)
    t_prefill_end = time.perf_counter()
    prefill_latency_ms = (t_prefill_end - t_prefill_start) * 1000.0

    # First token
    next_token = _next_token(logits)
    torch.cuda.synchronize(device)
    t_first_token = time.perf_counter()
    ttft_ms = (t_first_token - t_start) * 1000.0

    generated: list[int] = [next_token.item()]
    eos_id = tokenizer.eos_token_id

    # --- Decode ---
    decode_start = time.perf_counter()
    for _ in range(config.max_new_tokens - 1):
        if eos_id is not None and generated[-1] == eos_id:
            break

        if config.use_cache and past_key_values is not None:
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
        else:
            # Recompute full sequence when cache disabled
            full_ids = torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)
            outputs = model(input_ids=full_ids, use_cache=False)
            past_key_values = None

        logits = outputs.logits
        next_token = _next_token(logits)
        generated.append(next_token.item())

    torch.cuda.synchronize(device)
    decode_end = time.perf_counter()

    generated_token_count = len(generated)
    decode_total_ms = (decode_end - decode_start) * 1000.0
    decode_per_token_ms = decode_total_ms / max(generated_token_count, 1)
    decode_tps = (
        (generated_token_count / (decode_total_ms / 1000.0)) if decode_total_ms > 0 else 0.0
    )
    end_to_end_ms = (decode_end - t_start) * 1000.0

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    mem_after_prefill = torch.cuda.memory_allocated(device)
    kv_cache_mb = max(0.0, (mem_after_prefill - mem_before) / (1024 * 1024))

    all_ids = input_ids[0].tolist() + generated
    text = tokenizer.decode(all_ids, skip_special_tokens=True)

    timings = GenerationTimings(
        prompt_token_count=prompt_token_count,
        generated_token_count=generated_token_count,
        prefill_latency_ms=prefill_latency_ms,
        ttft_ms=ttft_ms,
        decode_latency_per_token_ms=decode_per_token_ms,
        decode_tokens_per_second=decode_tps,
        end_to_end_latency_ms=end_to_end_ms,
        peak_vram_mb=peak_vram_mb,
        kv_cache_memory_mb=kv_cache_mb,
    )
    return GenerationOutput(text=text, token_ids=generated, timings=timings)
