
# EdgeShield AI v0.4.0 — Architecture

## Product statement
A local-first cybersecurity intelligence system with three inputs (network, URL,
SMS), lightweight detection, local advisory, adaptive learning and a self-correcting
intelligence control plane.

## Scope contract
No external sensors are assumed or supported in this release. The three input
pipelines are:
- Network: uploaded PCAP/PCAPNG, optionally live packet/flow capture in a later adapter.
- URL: explicit URL submission and local URL feature/reputation analysis.
- SMS: explicit SMS/message text submission.

SoC telemetry (CPU, NPU, memory, temperature, runtime health) is internal platform
observability, not a fourth data-input pipeline.

## Logical architecture
1. Input adapters: network capture, URL submission, SMS submission.
2. Normalisation: PCAP validation/flow extraction, URL canonicalisation, message normalisation.
3. Detection: network, URL and SMS detectors; each emits a common ThreatEvent.
4. Threat reasoning: deterministic risk scoring, event correlation and optional Llama advisory.
5. Local application: FastAPI, dashboard, SQLite event store.
6. Adaptive learning: verified feedback, drift observations, candidate models and model registry.
7. SCI control plane: fault intake, diagnosis, allow-listed recovery recommendations, verification state and evidence ledger.
8. Platform observability: Arm Linux CPU/memory/thermal/runtime checks; optional NPU/driver adapter only when supported by selected SoC.

## SCI fault domains
- `network`: capture parsing, flow construction, feature schema, detector pipeline.
- `url`: parsing, canonicalisation, feature extraction, detector pipeline.
- `sms`: encoding, normalisation, feature extraction, detector pipeline.
- `model`: artifact integrity, output schema, disagreement, inference timeout.
- `soc`: memory pressure, thermal limit, accelerator failure, CPU fallback availability.
- `runtime`: service health, dependency errors, invalid input, software integrity.

## SCI lifecycle
DETECT → DIAGNOSE → CORRECT → PROVE

Detection sources:
- Pipeline exceptions automatically create an unverified SCI record.
- Future monitors should detect schema violations, invalid outputs, drift, timeouts,
  checksum failures, resource pressure and model disagreement.
- Semantic misclassification detection requires independent ground truth or
  validated human feedback; no system can infer every wrong decision from output alone.

Diagnosis:
- Current implementation: deterministic rule-based triage.
- Future: structured fault signatures, causal traces, model-specific diagnostics.

Correction:
- Current implementation: component-specific allow-listed recommendation is recorded.
- Future: safe recovery adapters for reparse/recompute, fallback backend, isolated
  restart, rollback or accelerator quarantine.
- No arbitrary command execution, automatic firmware changes or security-policy mutation.

Proof:
- Verification states: verified / failed / unverified.
- `verified` requires all supplied checks to pass and independent verification flag.
- Current independent flag is caller-provided, not attested. Production needs trusted
  validators, signed records and protected keys.
- SHA-256 hash chain detects record tampering relative to the local chain; it does
  not prove factual correctness or trusted time.

## Arm platform plan
- Baseline CPU execution first.
- Read CPU/memory/thermal telemetry from Linux interfaces where available.
- Add accelerator health only through a board-specific adapter.
- Keep Llama inference on ExecuTorch as required by challenge brief.
- Optional detector acceleration (e.g. RKNN/Hailo) remains a separate backend.
- Record device model, OS/kernel, runtime versions, thermal state and test conditions.

## API
- POST `/api/url/analyze`
- POST `/api/sms/analyze`
- POST `/api/network/analyze-pcap`
- POST `/api/advisory/explain`
- POST `/api/adaptive/feedback`
- POST `/api/adaptive/network/observe`
- POST `/api/adaptive/candidate/predict`
- POST `/api/adaptive/candidate/evaluate`
- GET `/api/adaptive/status`
- POST `/api/sci/incidents`
- GET `/api/sci/incidents`
- GET `/api/sci/ledger/verify`
- GET `/api/sci/capabilities`

## Maturity statement
This codebase is a development prototype. Detection engines are heuristic baselines.
Adaptive learning trains an isolated candidate text model. SCI records exceptions and
operator-reported faults, applies rule-based diagnosis and records recommendations;
it does not yet execute real hardware recovery. Llama+ExecuTorch is still a
model-specific scaffold and must be integrated and demonstrated on the selected Arm
board before claiming challenge compliance.
