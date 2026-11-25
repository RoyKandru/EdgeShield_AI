# SCI Engine — v1

SCI is a cross-cutting reliability control plane for EdgeShield's three input
pipelines and shared AI/runtime infrastructure. No external sensor interfaces
or sensor fault domain are included.

## Fault domains
| Domain | Signals | Candidate bounded actions |
|---|---|---|
| network | PCAP parse errors, truncated capture, flow schema mismatch | reparse capture, rebuild flow features, fallback, mark uncertain |
| url | URL normalisation mismatch, feature schema failure, detector timeout | reparse URL, recompute URL features, fallback, mark uncertain |
| sms | encoding/normalisation errors, feature schema mismatch | renormalise message, recompute text features, fallback, mark uncertain |
| model | artifact integrity, output schema, disagreement, inference timeout | retry validated preprocessing, fallback, mark uncertain, quarantine model |
| soc | memory pressure, thermal limit, accelerator fault | CPU fallback, reduce workload, quarantine accelerator, request inspection |
| runtime | service/dependency failure, invalid input, integrity mismatch | reject input, isolated restart, rollback, disable faulty backend |

## Lifecycle
1. Detect a fault through exception handling, invariants, output validation,
   drift monitoring, hardware telemetry or operator report.
2. Diagnose using evidence and deterministic fault-signature rules.
3. Select a component-specific allow-listed recovery recommendation.
4. Run independent post-correction checks and record verified/failed/unverified.
5. Feed reviewed outcomes into adaptive candidate evaluation, never silent model mutation.

## Current implementation
- `app/sci/engine.py`: diagnosis rules, allow-lists, verification status, hash-linked ledger.
- `app/sci/schemas.py`: typed SCI incident schema.
- API: POST `/api/sci/incidents`; GET `/api/sci/incidents`;
  GET `/api/sci/ledger/verify`; GET `/api/sci/capabilities`.
- URL/SMS/PCAP route exceptions automatically create unverified incidents.

## Proof limitations
The hash chain protects local record integrity against undetected modification
relative to the chain. It does not establish truth, trusted time, or independence
of supplied checks. The `independent` flag is caller-declared. Production should
use trusted validators, signed evidence, protected keys and hardware-backed
attestation where available.

## Recovery limitations
Current actions are recommendations recorded in the ledger; they are not executed.
No arbitrary command execution, firmware flashing, chip repair or autonomous
security-policy mutation. Device-specific recovery requires authenticated adapters,
interlocks, watchdogs, rate limits and tested rollback.

## Coverage limitation
No system can guarantee detection of every wrong decision. A fault may have no
observable symptom or independent ground truth. Unknown outcomes must remain
unverified and trigger fallback or human review.
