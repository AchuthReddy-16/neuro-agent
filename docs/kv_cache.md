# KV Cache in Autoregressive Transformer Inference

## What the KV cache stores

For every transformer layer, autoregressive inference caches the **Key** and
**Value** tensors of previously processed tokens so they do not need to be
recomputed during every decode step.

During **prefill**, the model processes the full prompt in one (or few)
forward passes and stores per-layer K/V tensors for each prompt position.

During **decode**, only the new token is projected to Q/K/V. The cache lets
attention read prior K/V instead of recomputing them from the full prefix.

## Memory scaling

KV cache memory grows roughly linearly with:

- number of layers
- sequence length (prompt + generated tokens)
- number of KV heads
- head dimension
- bytes per element (BF16 = 2 bytes)

Formula (per layer, both K and V):

```
2 × batch × seq_len × num_kv_heads × head_dim × sizeof(dtype)
```

## Experiments in this stage

| Experiment | `use_cache` | Purpose |
|------------|-------------|---------|
| A | `True` | Normal inference with KV cache |
| B | `False` | Recompute full sequence each step (if HF supports) |
| C | `True` | Context-length scaling at 512/1024/2048/4096/(8192) |

Results are saved under `benchmarks/kv_cache/`. Improvements are only
claimed when measured — see JSON output for actual numbers.

## Future optimization targets

Profiling results (PyTorch profiler, Nsight Systems, Nsight Compute) are
structured to identify whether custom kernels (RMSNorm, RoPE, attention,
quantized GEMM, activation fusion) would address a real bottleneck. No
custom CUDA/Triton kernels are written in this stage.
