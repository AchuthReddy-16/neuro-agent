# H.6 — Final Precision and Cost Comparison

Analysis/reporting only. Existing measured artifacts. RTX 4090 RunPod rate **$0.74/hour** (user-provided).

Model: Qwen3-4B-Instruct-2507 + `checkpoints/sft_corrected_v2/final`. Single-request, no concurrency.

## 1. Comparison table

| | A HF BF16 | B H.1B INT8 | C H.4 fair INT8 | D H.5 Triton INT8 | E vLLM BF16 | F vLLM FP8 (ref) |
|---|---:|---:|---:|---:|---:|---:|
| Quality | 0.864 | 0.866 | 0.875 | 0.875 | 0.938 | 0.938 |
| Quality n / protocol | 1000 full | 1000 full | 32 sanity | 32 sanity | 16 smoke | 16 smoke |
| Model VRAM (MB) | 7938.4 | 4479.0 | 4476.9 | 4691.4 | 7966.7 | 4720.6 |
| Peak VRAM (MB) | 8164.2 | 4709.3 | 4697.1 | 4911.6 | 8243.2 | 5181.4 |
| Load time (s) | 23.71 | 6.15 | 13.57 | 6.55 | 62.35 | 13.96 |
| TTFT (ms) | 51.05 | 112.50 | 79.30 | 105.67 | N/A | N/A |
| Decode tok/s | 52.74 | 12.50 | 18.73 | 15.60 | 61.30 | 54.10 |
| Latency/token (ms) | 18.96 | 84.70 | 53.41 | 64.11 | 16.30 | 18.50 |
| E2E (ms) | 1264.50 | 5450.00 | 3497.46 | 4208.68 | 1043.80 | 1182.50 |
| VRAM reduction vs A (%) | 0.00 | 42.32 | 42.47 | 39.84 | -0.97 | 36.53 |
| Throughput Δ vs A (%) | 0.00 | -76.30 | -64.49 | -70.42 | 16.23 | 2.58 |
| E2E latency Δ vs A (%) | 0.00 | 331.00 | 176.59 | 232.83 | -17.45 | -6.48 |
| Requests/hour (single-req E2E) | 2846.98 | 660.55 | 1029.32 | 855.38 | 3448.94 | 3044.40 |
| Generated tokens/hour (decode) | 189864.0 | 45000.0 | 67428.0 | 56160.0 | 220680.0 | 194760.0 |

Deltas vs **A HF BF16**. Negative VRAM reduction means more memory than BF16. Positive E2E Δ means slower.

FP8 is **not** INT8. Column F is an 8-bit floating-point vLLM reference only.

## 2. Cost table

Rate: **$0.74/GPU-hour**. Cost = rate × wall time implied by measured single-request E2E (requests) or decode tok/s (tokens). No batching.

| | A HF BF16 | B H.1B INT8 | C H.4 fair INT8 | D H.5 Triton INT8 | E vLLM BF16 | F vLLM FP8 (ref) |
|---|---:|---:|---:|---:|---:|---:|
| Cost / 1K requests (USD) | 0.259925 | 1.120278 | 0.718922 | 0.865118 | 0.214559 | 0.243069 |
| Cost / 1M generated tokens (USD) | 3.8975 | 16.4444 | 10.9747 | 13.1766 | 3.3533 | 3.7995 |

## 3. Key findings

1. **bnb INT8 saved memory but suffered backend-wide latency overhead.** Peak VRAM fell ~42% vs HF BF16 (~8.2 GB → ~4.7 GB), while H.1B decode dropped from 52.74 to 12.5 tok/s (−76%) and E2E rose from 1265 ms to 5450 ms (+331%).
2. **H.4 fair INT8 is not an optimized INT8 path.** Warning-suppressed rerun is 18.73 tok/s / 3497 ms E2E — better than H.1B timing artifacts, still far slower than BF16. FP16 casts, CUDA Graphs, and torch.compile were rejected.
3. **Triton is not a production win.** Isolated M=1 fuse was +29.5% on one hot linear; full-model generate() was 15.60 tok/s / 4209 ms E2E, worse than H.4 fair INT8.
4. **vLLM BF16 is the current low-latency serving baseline** among measured options: 61.3 tok/s, 1044 ms E2E (−17% vs HF BF16 E2E, +16% throughput).
5. **vLLM FP8 is a separate 8-bit floating-point reference**, not INT8: 54.1 tok/s, 1183 ms E2E, ~5.06 GB peak. Quality is 16-example smoke only.
6. **Cost follows latency.** At $0.74/h, 1K single-request jobs cost ~$0.26 on HF BF16 vs ~$1.12 on H.1B INT8 vs ~$0.21 on vLLM BF16. Token cost is ~$3.90/M on HF BF16 vs ~$16.44/M on H.1B INT8 vs ~$3.35/M on vLLM BF16.

## 4. Production implications

- Do **not** choose bitsandbytes INT8 for production serving on this GPU if latency or cost/request matters.
- Do **not** deploy the H.5 Triton patch.
- Use **vLLM BF16** as the measured low-latency serving baseline.
- Keep **FP8** as a memory-saving serving reference until a later production decision; do not treat it as INT8.
- H.6 does **not** start concurrency or load testing.

## 5. H.6 verdict

**PASS** — matrix and costs built from existing artifacts with the provided $0.74/hour rate. No new runs, no git mutation.

