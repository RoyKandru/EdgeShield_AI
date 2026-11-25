# EdgeShield AI v0.2.0 — Revision 2

## Added
- Local adaptive feedback ledger with verification and reviewer provenance.
- Verified-feedback-only incremental text candidate learner.
- EWMA network metric baselines and simple drift flags.
- Candidate metric gate and model registry metadata.
- Adaptive API endpoints and tests.

## Validation
- `pytest -q`: 10 tests passed in the build environment.

## Important boundaries
- Candidate learner is isolated from production detection.
- Candidate metric gate validates supplied metrics; it does not calculate them.
- Production promotion and automatic rollback are not wired.
- Production detectors remain heuristic baselines.
- Llama + ExecuTorch remains a model/device-specific scaffold pending end-to-end validation.
