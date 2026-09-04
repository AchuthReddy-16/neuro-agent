# Neuro-Agent FastAPI Contract

Base URL (local): `http://127.0.0.1:8080`

CORS allows `http://localhost:3000` / `3001` by default (`NEURO_API_CORS_ORIGINS`).

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness + model labels |
| GET | `/api/system/metrics` | Demo telemetry (nulls when unmeasured) |
| POST | `/api/upload` | Multipart upload → experiment |
| POST | `/api/analyze` | Run research agent pipeline |
| GET | `/api/experiment/{experiment_id}` | Experiment state |
| GET | `/api/visualization/{visualization_id}` | Image bytes or JSON metadata |

Aliases: `demo` / `exp_demo` → `exp_demo_s001` (processed sample `S001_R01_E000`).

---

## POST `/api/upload`

`multipart/form-data`

| Field | Required | Notes |
|-------|----------|-------|
| `file` | yes | Binary body |
| `fileType` | yes | `eeg` \| `figure` \| `metadata` |
| `filename` | recommended | Falls back to upload filename |
| `experiment_id` | no | Attach to existing experiment |

**Supported uploads (honest):**

- **figure**: `.png` / `.jpg` / `.jpeg` / `.webp`
- **eeg** / **metadata**: JSON only, must include `sample_id` that exists in the processed registry (e.g. `{"sample_id":"S001_R01_E000"}`). Raw EDF/CSV/NPY are **not** parsed.

**Response (camelCase ids):**

```json
{
  "experimentId": "exp_…",
  "assetId": "asset_…",
  "uploaded_artifacts": […],
  "detected_input_types": ["eeg", "metadata", "vision"],
  "available_visualizations": [{ "id", "tab", "title", "imageUrl", "index", … }],
  "metadata": { "subject", "run", "taskType", "movementCondition", "samplingRateHz", "channels", "sampleId" },
  "eeg": { "samplingRateHz", "channels", "channelLabels", "sampleId", "format", … },
  "status": "ready"
}
```

Errors: `400` missing/empty/unsupported/malformed; `413` too large; `404` unknown experiment.

---

## POST `/api/analyze`

Accepts **snake_case or camelCase** for experiment id:

```json
{ "experimentId": "exp_demo_s001", "question": "…", "imageId": null, "visualizationId": null, "context": null }
```

Runs: intent → tools → EvidenceBundle → grounded answer → conditional verifier → ≤1 recovery.
Vision VLM runs only when intent `requires_vision` (or related routing fields) **and** `NEURO_API_ENABLE_VLM=1`.

**Response** (evidence keys snake_case; nested timing/system/timeline camelCase for the web app):

```json
{
  "answer": "…",
  "route": "TEXT",
  "route_detail": {
    "intent": {…},
    "requires_vision": false,
    "requested_visual_type": null,
    "question_type": "…"
  },
  "computed_evidence": [
    { "label", "value", "unit", "tool", "metric", "channel", "band", "condition", "provenance", "highlight" }
  ],
  "visual_evidence": [
    { "id", "label", "tab", "observation", "imageUrl", "image_type", "vlm_interpretation", "provenance" }
  ],
  "model_interpretation": "…",
  "tools_used": ["…"],
  "verification": {
    "status": "passed|triggered|recovered|skipped",
    "message": null,
    "recoveryPerformed": false,
    "triggered": false,
    "result": null,
    "recovery_triggered": false
  },
  "uncertainty": "…",
  "timing": {
    "totalMs", "routingMs", "toolsMs", "visionMs", "synthesisMs",
    "generation_ms", "verificationMs", "verifier_ms", "recoveryMs"
  },
  "system": {
    "textModel", "visionModel", "precision", "serving", "route",
    "verifierStatus", "text_backend", "vision_backend", "serving_mode"
  },
  "timeline": [{ "id", "name", "status", "latencyMs", "summary" }],
  "question": "…",
  "id": "request_id",
  "raw_tool_output": "…",
  "experiment_id": "…"
}
```

Unavailable evidence fields are omitted or null — never fabricated.

Errors: `400` missing question / missing image for vision; `404` invalid experiment; `503` model/VLM unavailable; `500` tool failure (no stack traces).

---

## GET `/api/experiment/{id}`

Returns experiment record: metadata, eeg, visualizations, modalities, files, analysis_history, `isDemo`.

## GET `/api/visualization/{id}`

- `Accept: application/json` → visualization metadata JSON (`imageUrl`, tab, …)
- Otherwise → image file bytes when path exists

## GET `/api/health`

```json
{ "status": "ok", "version": "0.1.0", "backend": "fastapi", "textModel", "visionModel", "servingMode", "agentLoaded", "visionLoaded" }
```

## GET `/api/system/metrics`

Live GPU memory/util when `nvidia-smi` works; latency/route/verifier reflect **last request** only (null until first analyze). Precision label: `INT8 W8A8`. Do not treat null telemetry as zeros.

---

## Env knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `NEURO_API_CORS_ORIGINS` | localhost:3000,3001 | CORS allowlist |
| `NEURO_API_MAX_UPLOAD_MB` | 25 | Upload size cap |
| `NEURO_API_ENABLE_VLM` | 0 | Run HF+PEFT VLM when vision required |
| `NEURO_API_LOAD_AGENT` | 0 | Eager-load text agent on startup |
| `NEURO_SERVING_MODE` | hybrid | Metrics / system label |
| `NEURO_API_STORE_ROOT` | `results/api_experiments` | Experiment store |

## Run

```bash
PYTHONPATH=src .venv/bin/uvicorn neuro_agent.api.app:app --host 127.0.0.1 --port 8080
```
