# EdgeShield AI — C++-first target architecture

## Principle
The deployed edge product is C++-first. Python is retained for model research and engineering workflows, not as a mandatory dependency of the production detection path. Migration is incremental and contract-led; do not rewrite all modules before the challenge deadline.

## Layers
1. **Input adapters:** Network PCAP/PCAPNG, explicit URL, explicit SMS.
2. **C++ detection data plane:** validation, canonicalisation, feature extraction, independently deployable detector interfaces.
3. **Threat intelligence plane:** versioned ThreatEvent, evidence, risk fusion, correlation and bounded advisory context.
4. **SoC execution plane:** CPU baseline, mandatory Llama-family ExecuTorch advisory backend, optional RKNN backend for compatible detector models.
5. **Platform abstraction:** capability discovery, telemetry, resource admission, ARM64 portability and backend selection.
6. **SCI control plane:** Detect → Diagnose → Correct → Prove → Evolve; allow-listed reversible recovery only, approval gates and auditable evidence.
7. **Product services:** local API, dashboard, storage, updates, audit and deployment packaging.

## Current native foundation
- `native/include/edgeshield/contracts.hpp`: versioned event/evidence contract starter.
- `native/include/edgeshield/detectors.hpp`: text detector interface.
- `native/src/detectors.cpp`: URL/SMS lexical heuristic examples.
- `native/src/risk.cpp`: conservative starter fusion rule.
- `native/src/system_monitor.cpp`: Linux memory, root filesystem and thermal sample where exposed.
- CMake and native smoke tests.

## Porting order
1. Freeze event/schema and detector contracts; create Python↔C++ golden vectors.
2. Port URL/SMS feature extraction and detectors; compare outputs against labelled test vectors.
3. Port network flow extraction with bounded PCAP streaming and bidirectional flow handling.
4. Port event processing, risk fusion and persistence interfaces.
5. Port SCI policy engine and evidence ledger.
6. Integrate ExecuTorch Llama through its supported C++ API and validate actual `.pte` + tokenizer + model-specific runner on target.
7. Add RKNN C++ only for a suitable converted detector model; keep separate from mandatory Llama path.
8. Wire native telemetry to System Monitor UI and benchmark harness.

## Non-negotiable constraints
- Exactly three external analysis inputs; telemetry is internal observability.
- Offline inference; no URL fetching during classification.
- LLM explains supplied evidence; it does not classify authoritatively, invent facts, modify policy or execute commands.
- Risk score is not classifier confidence; heuristic confidence is not a calibrated probability.
- No unsupported accelerator utilisation percentages or unmeasured power claims.
- No arbitrary shell execution, destructive autonomous remediation or firmware flashing.
- No claims of challenge compliance without end-to-end target-board evidence.
