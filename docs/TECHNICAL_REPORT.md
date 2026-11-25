# EdgeShield AI (vv1) — Technical Engineering Report

**Bharat AI-SoC Challenge 2026–27 · Problem Statement 3: On-Device Threat, Phishing & Scam Detection**
**Target Hardware:** Rockchip RK3588J (Quad Cortex-A76 + Quad Cortex-A55 Arm64 CPU + 3-Core 6 TOPS RKNPU, 31 GB RAM, Debian/Arm64 Linux)

---

## 1. Executive Summary
EdgeShield AI is a **100% offline, zero-cloud Arm64 edge security gateway** engineered to protect LAN/IoT devices and connected users against **network intrusions**, **phishing URLs**, and **scam/smishing SMS** (including India-specific Digital Arrest, KYC/OTP, UPI, APK dropper, and Hindi/Hinglish fraud).

Unlike CPU-only linear prototypes, EdgeShield executes **strict dual-pipeline hardware isolation** on real RK3588J silicon:
- **Pipeline A (3-Core RK3588 NPU, FP16):** Three deep-learning detectors pinned one-per-core (`NPU_CORE_0`, `NPU_CORE_1`, `NPU_CORE_2`) achieving **`415.83 req/s`** combined concurrent HTTP throughput.
- **Pipeline B (Arm64 CPU, ExecuTorch XNNPACK + Arm KleidiAI):** Persistent native C++ daemon (`bin/executorch_runner`) running quantized **Llama-3.2-1B** (`llama_3_2_1b_xnnpack_ctx512.pte`, **`766 MB`**, **`max_context_len=512`**, `8da4w` `G=128` + 4-bit embedding quantization `G=32`) at **`26.54 tok/s`** sustained over 20 back-to-back runs (**`3449.0 ms`** median wall time, **`918.84 MB`** RSS, **`+0.00 MB`** delta, zero throttling), wrapped with a **prompt-injection sanitizer** and **verdict-locked bilingual (`EN / हिंदी`) output validator**.
- **Governance & Containment:** A **9-stage Closed-Loop Self-Correcting Intelligence (SCI)** engine with Golden Regression Suite gating, atomic policy versioning/rollback, a **Reversible Containment Ladder** (`nftables` / `dnsmasq` dry-run by default with TTL + one-click Undo + protected-host safeguards), SHA-256 hash-linked audit ledger, optional constant-time `X-EdgeShield-Token` API auth, and an opt-in SSRF-hardened Link Sandbox (strictly disabled in offline mode).

---

## 2. System Architecture & Hardware Partitioning

```
[LAN NIC enP2p33s0] --> bin/live_sniffer (AF_PACKET, cap_net_raw) --> FlowReducer (26 features)
                                                                   --> DNS/HTTP URL Extractor
                                                                   --> SIP/Wire SMS Extractor
[Android Phone LAN] --> scripts/android_sms_forwarder.py ----------+
                                                                   v
  +==================================================================================+
  |  PIPELINE A: 3-CORE RK3588 NPU (FP16 via RKNN-Toolkit-Lite2 2.3.2, 3 Workers)    |
  |  Core 0: NetworkDualNPU (1D-CNN+MLP)  -> network_ids_fp16.rknn  (123 KB)         |
  |  Core 1: URLDualBranchNPU (MLP+CNN)   -> url_hybrid_fp16.rknn   (219 KB)         |
  |  Core 2: SplitBiGRUExactPacked + Pool -> sms_bigru_fp16.rknn    (3.9 MB)         |
  +==================================================================================+
                                      |
            Deterministic Risk Score + Cross-Channel Correlator (Detectors Decide!)
                                      |
  +==================================================================================+
  |  PIPELINE B: ARM64 CPU (ExecuTorch XNNPACK + Arm KleidiAI, C++ Persistent Daemon)|
  |  Prompt-Injection Sanitizer -> bin/executorch_runner (Llama-3.2-1B, ctx=512,     |
  |  766 MB, 8da4w+emb4) -> Verdict-Locked Validator (EN/HI + Loop/Phone Guard)      |
  +==================================================================================+
                                      |
  +==================================================================================+
  |  GOVERNANCE: 9-Stage Closed-Loop SCI + Reversible Containment + SHA-256 Ledger   |
  |  Golden Suite Gate | Policy Versioning & Rollback | nftables/dnsmasq TTL + Undo  |
  +==================================================================================+
```

**Core Safety Rule — Detectors Decide, Llama Explains:** Pipeline B never computes risk verdicts, never modifies SCI detection policy (`llm_policy_modification_allowed = false`), and cannot override Pipeline A's classification.

---

## 3. Detector Architectures, Datasets & Held-Out Benchmarks

| NPU Core | Model & Reference | Training Datasets (Real Public + Edge) | Held-Out Test Metrics | RKNN FP16 Parity (`max_diff`) |
|---|---|---|---|---|
| **Core 0** | `NetworkDualNPU` (`1D-CNN 1×26→48` + `MLP 26→64→64`, 23K params, `signed_log1p + z-score`) | CIC-IDS2017 (6 attack days: DDoS, PortScan, DoS Hulk/Slowloris/Slowhttptest, Patator, Bot, Web) + Edge LAN (`289,981` flows) | ROC-AUC **1.0000**, F1 **0.9985** (@0.50), P **0.9999**, R **0.9971**, FPR **0.00%** | `0.000003` |
| **Core 1** | `URLDualBranchNPU` (25-feat Lexical MLP + Char 1D-CNN `61×128→64`, Prasad et al. 2026) | Domain-grouped Phishing Dataset + Tranco Top-35k (`160,000` URLs; disjoint OpenPhish & Tranco 50k–60k eval) | ROC-AUC **0.9781**, AP **0.9780**, P **0.9878** / FPR **0.80%** (@0.96); Live OpenPhish R **85.0%** (@0.50) | `0.000435` |
| **Core 2** | `SplitBiGRUExactPacked` + Masked Global Mean Pool + WBCE (Phap 2026; exact `pack_padded_sequence` equivalence `diff=2.2e-8`) | Deduplicated UCI SMS (5,574) + LSDST-2022 (5,971) + Indian Scam/Benign OTP (720) = `6,573` unique rows | ROC-AUC **0.9942**, AP **0.9884**, F1 **0.9592** (@0.50), Recall **0.9793** (`5:1` cost-sensitive τ=0.22) | `0.000245` |

---

## 4. Measured RK3588J Silicon Performance (All Raw JSONs in `benchmarks/results/`)

### 4.1 Simultaneous 3-Core NPU Concurrency (`rk3588_3core_npu_concurrency.json`)
- **NPU_CORE_0 (Network IDS):** **200.77 req/s** · NPU p50/p95: **0.643 / 1.232 ms**
- **NPU_CORE_1 (Phishing URL):** **143.29 req/s** · NPU p50/p95: **1.611 / 1.837 ms**
- **NPU_CORE_2 (SMS BiGRU):** **71.77 req/s** · NPU p50/p95: **8.653 / 9.102 ms**
- **Combined Simultaneous Throughput:** **`415.83 req/s`** · NPU thermal rise: `47.2°C → 49.0°C`

### 4.2 Sustained 20-Run ExecuTorch Llama-3.2-1B Thermal & Memory Proof (`llama_sustained_20run_thermal.json`)
- **Quantization & Context Upgrade:** Re-exported with `-qmode 8da4w -G 128 -E 4,32 --max_seq_length 512 --max_context_length 512`, reducing `.pte` size from `1.69 GB` → **`766 MB`** (`-54.7%`), warm daemon RSS from `1,681.07 MB` → **`918.84 MB`** (`-45.3%`), and expanding context `4×` (`128 → 512`).
- **Warm-Reuse Optimization:** Reusing the persistent `TextLLMRunner` via `runner->reset()` delivers **`26.54 tok/s`** decode (**`3449.0 ms`** median end-to-end wall time, max `5456.5 ms`) with zero cross-event KV-cache bleed.
- **Sustained Speed:** First 5 runs **26.72 tok/s** → Last 5 runs **27.68 tok/s** (overall **26.54 tok/s**, throttling=`False`).
- **Peak CPU Big-Core Temp:** **84.1°C** · **ExecuTorch RSS Delta:** `918.84 MB → 918.84 MB` (`+0.00 MB`, exact zero memory leak).

### 4.3 Per-Process Memory Footprint (`system_memory_footprint.json`)
- **Pipeline A (FastAPI + 3 RKNN NPU Workers + Live NIC Sniffer):** **359.41 MB**
- **Pipeline B (Warm ExecuTorch Llama-3.2-1B `ctx=512` Daemon):** **918.84 MB**
- **Total EdgeShield RSS:** **1278.25 MB** (4.02% of 31786.8 MB board RAM — fits comfortably even on 4 GB / 8 GB Arm edge devices)

### 4.4 Independent Adversarial Red-Team Audit (`adversarial_redteam_audit.json`) & Regression Suite (`sci_and_regression_proof.json`)
- **Live-Wire IP-Only Audit:** `10/10` attack families (`468` raw packets) → **`+114`** passive NIC detections.
- **160-Case Adversarial API Audit:** **`100.0%` Attack Recall** (`129/129`, **`0` FN**), Precision **`96.99%`**, F1 **`98.47%`**, Benign FPR **`12.9%`** (`4/31`), LAN p50 latency **`32.364 ms`**.
- **Automated Pytest Suite:** `61 passed, 3 skipped in 15.26s` · **SCI SHA-256 Ledger:** `valid=True` (`records_checked=8`).

---

## 5. Security Hardening, Governance & Known Limitations

### Safety & Governance Controls
- **Verdict-Locked Advisory Validator (`app/advisory/validator.py`):** Blocks prompt injection, rejects verdict contradictions, strips hallucinated phone numbers (except India helpline `1930`), catches repetition loops, and supplies incident-specific EN & Devanagari Hindi actions.
- **Reversible Containment (`app/sci/containment.py`):** Dry-run by default (`EDGESHIELD_CONTAINMENT_DRY_RUN=true`), auto-protects local/gateway/DNS hosts, attaches finite TTLs + one-click Undo, and hashes every apply/undo record into the SHA-256 SCI ledger.
- **API Authentication (`app/auth.py`):** Optional constant-time `X-EdgeShield-Token` / `Authorization: Bearer` enforcement (`EDGESHIELD_API_TOKEN`) protecting all mutating SCI, Containment, Monitor, and Adaptive routes.

### Honest Known Limitations & Failure Modes
1. **Encrypted Payloads (TLS / DoH / QUIC):** Passive `AF_PACKET` inspection sees flow metadata, ports, timing, and unencrypted DNS/HTTP/SIP, but cannot inspect inside TLS-encrypted payloads or DNS-over-HTTPS streams without MITM termination (intentionally not performed for privacy).
2. **Compromised Legitimate Domains / Shorteners:** In strict offline mode (`EDGESHIELD_SANDBOX_ENABLED=false`), URL scoring inspects the URL string & lexical/CharCNN features only; phishing hosted on a clean compromised domain requires enabling the opt-in SSRF-safe Link Sandbox.
3. **Base Checkpoint Bilingual Hindi Mode (`ctx=512`, `766 MB`):** The deployed `llama_3_2_1b_xnnpack_ctx512.pte` (`8da4w` + 4-bit embeddings, `max_context_len=512`, `918.84 MB` RSS) is exported from the local `Llama-3.2-1B` base weights; Hindi mode pairs live on-device Hinglish narratives from Llama (`27.1 tok/s`) with full Devanagari (`हिंदी`) incident-specific vetted recommendations and grounded fallback. Swapping in an instruction-tuned Hindi checkpoint using `scripts/export_llama_executorch.sh` requires zero C++ changes.
4. **Gateway vs. Smartphone Form Factor:** Cellular SMS on personal phones does not traverse LAN Ethernet unless forwarded via the companion Termux/webhook script (`scripts/android_sms_forwarder.py`).
5. **Benign Edge-Case False Positives:** On the aggressive 160-case adversarial red-team audit, 4 of 31 benign control cases (`12.9%`) triggered conservative alerts; operators can resolve recurring benign patterns cleanly via the 9-stage SCI closed-loop allowlist with zero-regression Golden Suite verification.
