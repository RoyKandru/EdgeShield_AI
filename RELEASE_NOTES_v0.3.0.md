# EdgeShield AI v0.3.0 — SCI Engine

Added:
- Cross-component Self-Correcting Intelligence (SCI) incident workflow.
- Rule-based diagnosis for model, sensor, SoC and software fault reports.
- Component-specific allow-listed correction recommendations.
- Verified / failed / unverified post-correction status.
- SHA-256 hash-linked local evidence ledger and integrity verification endpoint.
- SCI tests covering verification, allow-lists and tamper detection.

Validation: 14 tests passed in the build environment.

Limitations:
- No universal guarantee that every fault can be detected.
- Recovery actions are recorded as recommendations, not executed.
- No physical sensor recalibration, chip repair or firmware recovery.
- Hash chain protects local record integrity but does not prove factual correctness or trusted time.
- Actual Llama+ExecuTorch integration and trained production detectors remain pending.
