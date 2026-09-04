# I.1 — W8A8 INT8 online concurrency (prefix cache off)

**Verdict: PASS**

Checkpoint: `checkpoints/text_w8a8_int8_compressed`  
Serving: vLLM 0.28 online OpenAI server, compressed-tensors W8A8, `CutlassInt8ScaledMMLinearKernel`  
Workload: ~512 prompt (measured **572** tokens) / **64** gen, `ignore_eos=true`, greedy, same at every level  
Prefix cache: **off** (`enable_prefix_caching=False`)  
Concurrency 32: **not tested** (peak nvidia-smi VRAM 22536 MB vs 24 GB gate)

## Scaling (vs this run’s concurrency=1)

| C | tok/s | vs C=1 | RPS | TTFT p50 / p95 (ms) | E2E p50 / p95 (ms) | vs C=1 E2E p50 | GPU % | fail |
|---|-------|--------|-----|---------------------|--------------------|----------------|-------|------|
| 1 | 131.38 | 1.00× | 2.05 | 27.4 / 29.2 | 484.9 / 486.5 | 1.00× | 99.4 | 0 |
| 4 | 447.47 | 3.41× | 6.99 | 80.5 / 85.5 | 568.5 / 576.5 | 1.17× | 99.6 | 0 |
| 8 | 765.88 | 5.83× | 11.97 | 122.6 / 153.8 | 667.9 / 674.3 | 1.38× | 100 | 0 |
| 16 | 1176.01 | 8.95× | 18.45 | 173.2 / 282.4 | 853.4 / 923.5 | 1.76× | 100 | 0 |
| 32 | — | — | — | — | — | — | — | skipped |

Throughput still scales through C=16 (sublinear after C=4). GPU util is already ~99% at C=1. Waiting queue appears at C=8 (max waiting 2) and C=16 (max waiting 7). Practical saturation for this workload: **16**. No SLA chosen.

nvidia-smi ~22.5 GiB is KV reservation at `gpu_memory_utilization=0.90`, not model weights. EngineCore model load: **4.27 GiB**. KV: 118,464 tokens (~28.9× at 4096).

## Single-request references (not the scaling baseline)

| Path | tok/s | E2E (ms) |
|------|-------|----------|
| HF BF16 | 52.74 | 1264.5 |
| HF bitsandbytes INT8 | 12.5 | 5450 |
| vLLM BF16 | 61.3 | 1043.8 |
| H.7 W8A8 offline `LLM.generate` | 134.3 | 476.5 |
| This run online C=1 | **131.38** | **484.9** |

## Continuous batching

Staggered HTTP streaming (4×64 @ t=0, 4×16 @ 200 ms, 2×32 @ 450 ms): shorts finished while longs were still active; later waves joined in-flight decode; `vllm:num_requests_running` peaked at 8. **Supported by evidence.**

Artifacts: `results/serving/load/w8a8_int8/`, `results/model_comparison/w8a8_int8_concurrency_scaling.json`.
