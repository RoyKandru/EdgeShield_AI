# EdgeShield AI v0.4.0

## Product scope
- Exactly three external input pipelines: network, URL and SMS.
- No external sensors or sensor-specific APIs, schemas, actions or adapters.
- CPU/NPU/memory/thermal/runtime are internal SoC/platform telemetry.

## SCI
- Fault domains: network, url, sms, model, soc, runtime.
- Component-specific recovery recommendation allow-lists.
- URL/SMS/PCAP processing exceptions automatically create unverified SCI records.
- Independent verification is required for a `verified` status.
- SHA-256 hash-linked SQLite evidence ledger and integrity verification endpoint.

## Adaptive learning
- Verified URL/SMS feedback can train an isolated candidate text learner.
- Network observations update local EWMA baselines.
- No silent production model promotion.

## Validation
- Python compilation passed.
- 16 automated tests passed.

## Known limitations
- Detectors are heuristic baselines pending trained-model integration.
- SCI correction actions are recorded recommendations, not executed recovery.
- Hardware telemetry/recovery adapters are not fully implemented.
- Independent verification is caller-declared, not cryptographically attested.
- Llama + ExecuTorch remains a model-specific scaffold pending target-device validation.
