# EdgeShield AI — High-Level Design v1

## Objective
Deliver an offline-capable edge cybersecurity product for ARM64 SoCs, with independent Network, URL and SMS analysis channels, unified threat events, SCI incident management and device observability. The Bharat AI SoC Challenge is the first validation milestone; the platform is intended to become a reusable EdgeProc product foundation.

## Major subsystems

1. **Input and ingestion:** URL/message API inputs, normalized network-flow input and classic PCAP file ingestion.
2. **Detection engines:** independent C++ detector interfaces for URL, SMS and network flows. Current implementations are heuristic baselines.
3. **Intelligence core:** common ThreatEvent contract, evidence, severity, risk score and cross-channel fusion.
4. **SCI:** record diagnosis, bounded correction recommendation, verification state and hash-linked ledger. No automatic destructive remediation.
5. **Device monitor:** Linux CPU, memory, storage, thermal, load, uptime and interface byte counters.
6. **Inference backends:** CPU baseline now; RKNN C++ and mandatory Llama + ExecuTorch C++ integration are separate planned adapters.
7. **Application:** native C++ HTTP API and browser dashboard.

## Quality attributes

- Local-first/offline operation; no cloud service is required by the native runtime.
- C++20 core and CMake build.
- Detector isolation through stable contracts.
- Hardware abstraction prevents RK3588 details leaking into the threat-event schema.
- Telemetry fields remain unavailable when the OS/driver does not expose them.
- API binds to loopback by default; TLS/authentication are deployment hardening requirements.

## Current scope boundary

This release does not claim trained model accuracy, mandatory ExecuTorch completion, NPU/GPU utilisation, live network capture or automatic blocking. See README for the exact capability matrix.
