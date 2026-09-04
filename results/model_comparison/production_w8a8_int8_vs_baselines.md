# H.7H — Production W8A8 INT8 vs baselines

**Verdict: PARTIAL PASS.** vLLM compressed-tensors W8A8 INT8 is a viable production INT8 candidate. Quality is close overall but `execution_vs_imagery` drops materially.

bitsandbytes INT8 remains rejected. FP8 is reference only and is not INT8.

## Comparison

| Metric | A HF BF16 | B bnb INT8 | C vLLM BF16 | D vLLM W8A8 INT8 | F vLLM FP8 (ref) |
|---|---:|---:|---:|---:|---:|
| Full quality | **0.864** (n=1000) | 0.866 (n=1000) | 0.938 smoke n=16 | **0.843** (n=1000) | 0.938 smoke n=16 |
| Peak VRAM | 8.16 GB | 4.71 GB | 8.24 GB | **5.39 GB** | 5.18 GB |
| VRAM reduction vs A | — | −42.3% | +1.0% | **−34.1% peak / −46.2% weights** | −36.5% |
| TTFT | 51.05 ms | 112.5 ms | N/A (offline) | N/A (offline) | N/A |
| Decode tok/s | 52.74 | 12.5 | 61.3 | **134.3** | 54.1 |
| E2E (64 tok) | 1264.5 ms | 5450 ms | 1043.8 ms | **476.5 ms** | 1182.5 ms |
| Throughput Δ vs A | — | −76.3% | +16.2% | **+154.6%** | +2.6% |
| Load time | 23.71 s | 6.15 s | 62.35 s | **14.22 s** | 13.96 s |
| Checkpoint size | 8.06 GB merged | n/a (runtime PTQ) | 8.06 GB + LoRA | **4.12 GB** | n/a |
| Kernel | HF BF16 Linear | Linear8bitLt | vLLM BF16 + LoRA | **CutlassInt8ScaledMMLinearKernel** | FP8 scaled MM |

## Per-task quality (n=1000, 125 each)

| Family | BF16 | W8A8 | Δ |
|---|---:|---:|---:|
| numerical_reasoning | 1.000 | 1.000 | 0 |
| statistical_comparison | 1.000 | 1.000 | 0 |
| tool_selection | 1.000 | 1.000 | 0 |
| band_power_analysis | 0.880 | 0.888 | +0.008 |
| channel_ranking | 0.816 | 0.832 | +0.016 |
| movement_task_classification | 0.728 | 0.728 | 0 |
| factual_grounding | 0.488 | 0.512 | +0.024 |
| execution_vs_imagery | 1.000 | **0.784** | **−0.216** |
| **overall** | **0.864** | **0.843** | **−0.021** |

Invalid output rate: 0.0 (no formatting regression). Avg output tokens: 7.72 vs BF16 7.86.

## Decision answers

1. **Quality:** Mostly preserved (−2.1 pp overall). Documented material drop on execution_vs_imagery.
2. **VRAM:** Weight footprint ~45–46% below BF16. Peak ~34% below HF BF16 peak.
3. **vs bnb INT8:** Substantially faster (134 vs 12.5 tok/s).
4. **vs vLLM BF16:** Faster on this contract (134 vs 61 tok/s); BF16 path still used runtime LoRA.
5. **Production INT8 candidate:** Yes, with the execution_vs_imagery caveat.
6. **Bottleneck:** Family-level quantization sensitivity, not MatMul8bitLt. Serving VRAM is KV-cache reservation.
7. **Further kernel work:** **NO.**

## What was not started

Concurrency/load, prefix caching, SLA, multimodal, frontend.
