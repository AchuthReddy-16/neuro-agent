# G.3A Primary Research Agent Evaluation Report

## 1. Architecture
User Question → PrimaryResearchAgent → Intent Selection (SFT model) → validate_intent → route_research_request → EvidenceBundle → Grounded Answer (SFT model).

## 2. Model / Checkpoint
- Base: Qwen/Qwen3-4B-Instruct-2507
- Adapter: checkpoints/sft_corrected_v2/final

## 3. Supported Intents
band_power, rms, psd_peak, channel_ranking, threshold_set, condition_comparison

## 4. Intent Parsing Success
- Schema validity: 100.0%
- Intent accuracy vs expected: 100.0%
- Valid JSON rate: 100.0%

## 5. Tool Selection / Execution
- Tool execution success: 100.0%
- Intent families covered: 6/6 (['band_power', 'channel_ranking', 'condition_comparison', 'psd_peak', 'rms', 'threshold_set'])

## 6. Avg Tool Calls
1.12

## 7. Max Tool Calls
2

## 8. Grounding Results
- Grounding pass rate: 100.0%
- Unsupported numeric claims: 0

## 9. Latency
- Avg: 5558.7 ms
- Max: 17689.5 ms

## 10. Peak VRAM
13387.3 MB

## 11. Success Trace Example
ID: g3a_001
Question: What is the beta band power at channel C3 for sample S001_R01_E000?
Intent: {
  "question_type": "band_power",
  "sample_id": "S001_R01_E000",
  "subject_id": null,
  "run_id": null,
  "epoch": null,
  "channels": [
    "C3"
  ],
  "frequency_band": "beta",
  "frequency_range": null,
  "metric": null,
  "condition_a": null,
  "condition_b": null,
  "threshold": null,
  "threshold_mode": "absolute",
  "comparator": null,
  "top_k": null,
  "sort_direction": "descending",
  "requested_visual_type": null,
  "image_id": null,
  "include_vision_evidence": false,
  "extra": {}
}
Answer excerpt: The computed value is 134.54469.

## 12. Failure Trace Example
None

## 13. Gate Result
**PASS**

## 14. Blockers
None

## 15. Ready for Verifier / Recovery?
Yes

E2E success rate: 100.0%
Failure categories: {}