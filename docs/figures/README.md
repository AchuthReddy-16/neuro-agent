# Benchmark Figures

Reproducible plots generated from **existing measured artifacts** under `results/**`.

```bash
python scripts/generate_benchmark_figures.py
```

Formats: PNG + SVG (200 DPI). Style: restrained scientific palette for GitHub README / reports.

| Figure | Files | Stage | Metric source | Interpretation |
|---|---|---|---|---|
| `00_headline_summary` | `.png` / `.svg` | headline summary | `Aggregated from base/sft_corrected_v2, production_w8a8, residency optimization, routing evaluation artifacts` | Five headline metrics for README top: text quality, W8A8 memory, serving speed, vision swap, routing. |
| `01_post_training_quality` | `.png` / `.svg` | text post-training quality | `results/{base_model_eval,sft_model_eval,sft_corrected_v2_eval,rlvr_model_eval}/summary.json; quantization/w8a8_int8/full_quality_eval.json` | Corrected SFT v2 peaks at 86.4%; W8A8 keeps 84.3% overall with a known category regression. |
| `02_category_quality_breakdown` | `.png` / `.svg` | W8A8 category quality | `sft_corrected_v2_eval/per_task_metrics.json; quantization/w8a8_int8/full_quality_eval.json` | Shows honest W8A8 tradeoff: execution_vs_imagery drops 100%→78.4% while other families hold. |
| `03_precision_serving_tradeoff` | `.png` / `.svg` | precision / serving tradeoff | `final_precision_cost_matrix.json; production_w8a8_int8_vs_baselines.json; text_quantization_bf16_int8_int4.json` | W8A8 sits in the high-throughput / mid-VRAM corner; bnb INT8 and Triton are memory-bound losers on speed. |
| `04_cost_comparison` | `.png` / `.svg` | cost comparison | `final_precision_cost_matrix.json; W8A8 costs derived with cost-matrix formulas from measured e2e/decode` | bnb INT8 is far more expensive per request/token; W8A8 is cheapest among measured INT8-class paths. |
| `05_concurrency_scaling` | `.png` / `.svg` | concurrency scaling | `model_comparison/w8a8_int8_concurrency_scaling.json` | Throughput scales to c16 (1176 tok/s); c32 skipped for VRAM safety near 24 GB. |
| `06_prefix_cache_impact` | `.png` / `.svg` | prefix cache | `serving/prefix_cache/w8a8_int8/concurrency_cache_comparison.json; w8a8_prefix_cache_comparison.json` | Warm prefix cache cuts TTFT sharply; decode throughput barely changes. |
| `07_sla_admission_control` | `.png` / `.svg` | SLA / admission | `model_comparison/w8a8_sla_admission_comparison.json` | Admission is overload protection, not a free latency win under already-good cached p95. |
| `08_routing_quality` | `.png` / `.svg` | routing evaluation | `routing/j1_summary.json; routing/confusion_matrix.json` | One prompt repair lifts routing accuracy 67.3%→99.0% and vision recall 41.2%→98.0%. |
| `09_vision_bottleneck_breakdown` | `.png` / `.svg` | end-to-end profiling | `model_comparison/final_end_to_end_profile.json (mean of vision E+F)` | VLM load + text restore dominate end-to-end vision latency (~56 s of swap overhead). |
| `10_k2_swap_vs_coresident` | `.png` / `.svg` | residency optimization | `model_comparison/model_swap_vs_co_resident.json` | Warm co-resident mode zeros swap overhead; hybrid policy retained because util=0.40 cuts KV capacity. |
| `11_kernel_investigation` | `.png` / `.svg` | kernel investigation | `optimization/int8_kernel/microbenchmark.json; int8_bnb_vs_triton_kernel.json` | Triton microkernel +29.5% faster, but model decode did not improve — not deployed. |
| `12_agent_reliability` | `.png` / `.svg` | agent reliability | `agent_primary_eval/summary.json; agent_recovery_eval/summary.json` | Primary agent is fully reliable on schema/tools/E2E; recovery reaches 96% overall E2E. |
| `13_multimodal_quality` | `.png` / `.svg` | multimodal SFT/RLVR | `model_comparison/multimodal_base_vs_sft_vs_corrected_vs_rlvr.json` | Corrected multimodal SFT (49.3%) is the selected VLM checkpoint over base/RLVR. |
| `14_latency_distributions_boxplot` | `.png` / `.svg` | concurrency scaling optional | `serving/load/w8a8_int8/per_request_traces.jsonl (timed requests only)` | Raw TTFT/E2E distributions by concurrency; not fabricated from summary percentiles. |

## Notes

- No invented numbers; values are read from JSON/JSONL result files.
- W8A8 cost bars are **derived** with the cost-matrix formulas from measured e2e/decode (not present in the measured cost matrix).
- bnb INT4 is marked **quality rejected** (targeted factual_grounding gate failed; full 1000-eval skipped).
- kernel investigation Triton is shown as a systems negative result, not a deployed optimization.
- residency optimization co-residency is not claimed as the universal production config; policy is **HYBRID**.
- Boxplots use raw per-request traces only.
