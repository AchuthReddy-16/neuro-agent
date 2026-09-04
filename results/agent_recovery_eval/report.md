# G.3B Conditional Verifier + Recovery Agent Evaluation Report

## 1. Architecture
User Question → PrimaryResearchAgent → Intent → Tools → Draft Answer → Deterministic Checks → (conditional) Verifier Model → (if fail) Recovery (max 1) → Re-verify → Final Answer.

## 2. Model / Checkpoint
- Base: Qwen/Qwen3-4B-Instruct-2507
- Adapter: checkpoints/sft_corrected_v2/final

## 3. Verification Schema
VerificationResult: passed, confidence_score, failure_codes, unsupported_claims, evidence_conflicts, missing_evidence, unit_issues, condition_mismatch, recommendation.

## 4. Deterministic Checks
Numeric grounding, channel existence, units, conditions, tool success, required evidence, conflicting numerics, tool loops, answer sections.

## 5. Trigger Policy
- Verifier trigger rate: 40.0%
- Deterministic-only acceptance: 0.0%

## 6. First-Pass Acceptance
0.0%

## 7. Verifier Model Call Rate
40.0%

## 8. Recovery Metrics
- Recovery rate: 40.0% (10 attempts)
- Recovery success: 100.0%
- Corruption recovery success: 72.7%

## 9. E2E Success
- Overall: 96.0%
- G.3A clean subset: 100.0%

## 10. Unsupported Claims
0

## 11. False Rejections (normal_easy)
0

## 12. Tool / Model Calls
- Avg tool calls: 1.08
- Max tool calls: 2
- Avg model calls: 2.76

## 13. Latency (NORMAL vs RECOVERY)
- Avg NORMAL path: 5543.8 ms
- Avg RECOVERY path: 13309.1 ms

## 14. Peak VRAM
13387.9 MB

## 15. Normal-Path Trace Example
ID: g3b_001
Verifier triggered: False
Path: NORMAL
Latency: 4742.6 ms

## 16. Recovery Trace Example
ID: g3b_007
Recovery action: REWRITE
Recovery success: True

## 17. Success Trace Example
ID: g3b_001
Answer excerpt: Answer: beta band power at C3 is 134.54469299316406 uV2
Evidence: channel=C3, band=beta, value=134.54469299316406 uV2
Tools used: compute_band_power
Uncertainty: Band power production uses source='fea

## 18. Categories
{
  "normal_easy": 10,
  "multi_tool": 2,
  "visual_ref": 1,
  "corrupted_draft_numeric": 3,
  "unsupported_channel": 2,
  "unit_mismatch": 2,
  "missing_channel": 1,
  "wrong_condition": 1,
  "insufficient_evidence": 1,
  "tool_param_mismatch": 1,
  "missing_sections": 1
}

## 19. Gate Result
**PASS**

## 20. Blockers
None

## 21. Intent Accuracy
100.0%

## 22. Traces Location
results/agent_recovery_eval/traces/

## Git Status
```
M .gitignore
 M pyproject.toml
 M src/neuro_agent/config.py
 M src/neuro_agent/inference/config.py
 M src/neuro_agent/inference/model_loader.py
 M src/neuro_agent/tools/__init__.py
 M src/neuro_agent/training/__init__.py
?? README_VISION_DATA.md
?? configs/eval.yaml
?? configs/eval_rlvr.yaml
?? configs/eval_sft.yaml
?? configs/eval_sft_corrected.yaml
?? configs/eval_sft_corrected_v2.yaml
?? configs/multimodal_eval.yaml
?? configs/multimodal_rlvr.yaml
?? configs/multimodal_sft.yaml
?? configs/multimodal_sft_corrective.yaml
?? configs/rlvr.yaml
?? configs/sft.yaml
?? configs/sft_corrective.yaml
?? configs/sft_corrective_v2.yaml
?? data/
?? docs/tool_layer_design.md
?? results/base_model_eval/
?? results/baseline_classifier/
?? scripts/analyze_multimodal_sft_regression.py
?? scripts/analyze_sft_regression.py
?? scripts/audit_multimodal_rlvr_data.py
?? scripts/audit_multimodal_training_data.py
?? scripts/build_multimodal_corrective_dataset.py
?? scripts/build_sft_corrective_v2_mixed_dataset.py
?? scripts/compare_model_evals.py
?? scripts/compare_multimodal_evals.py
?? scripts/estimate_format_recovery.py
?? scripts/inspect_rlvr_data.py
?? scripts/run_base_model_eval.py
?? scripts/run_eeg_baseline.py
?? scripts/run_multimodal_base_eval.py
?? scripts/run_multimodal_corrected_eval.py
?? scripts/run_multimodal_format_ablation.py
?? scripts/run_multimodal_rlvr_eval.py
?? scripts/run_multimodal_rlvr_targeted_gate.py
?? scripts/run_multimodal_rlvr_training.py
?? scripts/run_multimodal_sft_eval.py
?? scripts/run_multimodal_sft_training.py
?? scripts/run_multimodal_targeted_gate.py
?? scripts/run_primary_agent_eval.py
?? scripts/run_recovery_agent_eval.py
?? scripts/run_rlvr_eval.py
?? scripts/run_rlvr_training.py
?? scripts/run_sft_corrected_eval.py
?? scripts/run_sft_corrected_v2_eval.py
?? scripts/run_sft_corrective_training.py
?? scripts/run_sft_model_eval.py
?? scripts/run_sft_training.py
?? scripts/vision_compatibility_report.py
?? src/neuro_agent/agent/
?? src/neuro_agent/data/eeg_baseline.py
?? src/neuro_agent/evaluation/eeg_classifier.py
?? src/neuro_agent/evaluation/llm_eval.py
?? src/neuro_agent/evaluation/verifiers.py
?? src/neuro_agent/multimodal/
?? src/neuro_agent/tools/_stores.py
?? src/neuro_agent/tools/comparison.py
?? src/neuro_agent/tools/eeg_signal.py
?? src/neuro_agent/tools/evidence.py
?? src/neuro_agent/tools/metadata.py
?? src/neuro_agent/tools/normalization.py
?? src/neuro_agent/tools/ranking.py
?? src/neuro_agent/tools/router.py
?? src/neuro_agent/tools/schemas.py
?? src/neuro_agent/tools/vision_evidence.py
?? src/neuro_agent/training/dataset.py
?? src/neuro_agent/training/rewards.py
?? src/neuro_agent/training/rlvr_trainer.py
?? src/neuro_agent/training/trainer.py
?? tests/
?? web/
```

## Git Diff Stat
```
.gitignore                                | 34 +++++++++++-
 pyproject.toml                            |  6 +++
 src/neuro_agent/config.py                 |  4 ++
 src/neuro_agent/inference/config.py       |  1 +
 src/neuro_agent/inference/model_loader.py |  8 +++
 src/neuro_agent/tools/__init__.py         | 89 ++++++++++++++++++-------------
 src/neuro_agent/training/__init__.py      | 43 ++-------------
 7 files changed, 107 insertions(+), 78 deletions(-)
```
