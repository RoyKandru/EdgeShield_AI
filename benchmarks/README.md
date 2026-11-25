# EdgeShield AI — Verified RK3588J Hardware & Model Benchmarks

Every metric below was measured directly on the target **Arm RK3588J** board and backed by raw JSON outputs in [`results/`](results/).

**Reproduce on the board:**
```bash
cd source/edgeshield-ai
python3 scripts/run_benchmarks.py
```

---

## 1. Concurrent 3-Core RK3588 NPU Throughput & Latency (Live Silicon)
Raw log: [`results/rk3588_3core_npu_concurrency.json`](results/rk3588_3core_npu_concurrency.json)

| NPU Core | Detector | Throughput (req/s) | NPU Call p50 / p95 (ms) | HTTP E2E p50 / p95 (ms) | Live Core Load Under Burst |
|---|---|---:|---:|---:|---:|
| **NPU_CORE_0** | Network IDS (`network_ids_fp16.rknn`) | **200.77** | 0.643 / 1.232 | 4.755 / 6.996 | 15.1% |
| **NPU_CORE_1** | Phishing URL (`url_hybrid_fp16.rknn`) | **143.29** | 1.611 / 1.837 | 6.927 / 8.796 | 23.3% |
| **NPU_CORE_2** | SMS Smishing (`sms_bigru_fp16.rknn`) | **71.77** | 8.653 / 9.102 | 13.883 / 16.1 | 60.5% |
| **Combined (All 3 Cores Simultaneous)** | — | **415.83 req/s** | — | — | NPU Temp: 47.2°C → 49.0°C |

---

## 2. On-Device ExecuTorch Llama-3.2-1B-Instruct (EN + HI + Prompt-Injection + 20-Run Thermal Proof)
Raw logs: [`results/executorch_llama_arm64_bench.json`](results/executorch_llama_arm64_bench.json) · [`results/llama_sustained_20run_thermal.json`](results/llama_sustained_20run_thermal.json)

- **Model / Runtime:** `Llama-3.2-1B-Instruct (XNNPACK)` via `executorch-xnnpack-kleidiai-arm64` (daemon cold load: `4073.6 ms`)
- **4-Case Bilingual & Security Suite:** Mean Decode **26.55 tok/s** · Mean Wall **3946.8 ms** · `all_validated=True` · `prompt_injection_blocked=True`
- **20-Run Sustained Thermal & Leak Proof:**
  - First 5 runs vs. Last 5 runs: **26.72 → 27.68 tok/s** (overall mean **26.54 tok/s**, throttling=`False`)
  - Latency p50 / Max: **3449.0 ms / 5456.5 ms**
  - Peak Big-Core CPU Temp: **84.1°C**
  - ExecuTorch Daemon RSS Delta across 20 runs: **918.84 MB → 918.84 MB (`+0.00 MB`, zero memory leak)**

---

## 3. RK3588J Per-Process Memory Footprint (RSS)
Raw log: [`results/system_memory_footprint.json`](results/system_memory_footprint.json)

- **Pipeline A (FastAPI + 3 RKNN NPU Workers + Live NIC Sniffer):** **359.41 MB**
- **Pipeline B (Warm ExecuTorch Llama-3.2-1B Daemon):** **918.84 MB**
- **Total EdgeShield Footprint:** **1278.25 MB** (4.02% of 31786.8 MB board RAM)

| Component Process | PID | Resident RAM (RSS) |
|---|---:|---:|
| `executorch_llama_daemon` | 1871409 | **915.77 MB** |
| `uvicorn_api_server` | 1871376 | **232.32 MB** |
| `npu_core_2_sms_worker` | 1871464 | **49.52 MB** |
| `npu_core_1_url_worker` | 1871448 | **38.93 MB** |
| `npu_core_0_network_worker` | 1871424 | **37.89 MB** |
| `executorch_llama_daemon` | 1871332 | **3.07 MB** |
| `live_nic_sniffer` | 1871407 | **0.75 MB** |

---

## 4. Held-Out Model Accuracy & RKNN FP16 Parity
Raw log: [`results/detectors_heldout_metrics.json`](results/detectors_heldout_metrics.json)

- **Core 0 — Network IDS (`NetworkDualNPU`, 289,981 CIC-IDS2017 + Edge LAN flows):** Test ROC-AUC **1.0000**, F1 **0.9985** (@0.50), Precision **0.9999**, Recall **0.9971**, FPR **0.00%**, RKNN FP16 `max_diff = 0.000003`.
- **Core 1 — Phishing URL (`URLDualBranchNPU`, 160,000 domain-grouped Phishing + Tranco URLs):** Test ROC-AUC **0.9781**, AP **0.9780**, Precision **0.9878** / FPR **0.80%** (@0.96), RKNN FP16 `max_diff = 0.000435`.
- **Core 2 — SMS Smishing (`SplitBiGRUExactPacked + MaskedPool + WBCE`, 6,573 deduplicated UCI + LSDST + Indian SMS):** Test ROC-AUC **0.9942**, AP **0.9884**, F1 **0.9592** (@0.50), Recall **0.9793** (`5:1`, @0.22), RKNN FP16 `max_diff = 0.000245`.

---

## 5. Automated Test Suite, 9-Stage SCI & Containment Verification
Raw log: [`results/sci_and_regression_proof.json`](results/sci_and_regression_proof.json)

- **Pytest Suite:** `61 passed, 3 skipped in 15.26s` (`exit_code=0`, `20.53 s`)
- **SCI Hash-Linked Ledger:** `valid=True`, `records_checked=8`, `head_hash=4b9fcfa617a8444e...`
- **Reversible Containment Mode:** `dry_run` (`protected_hosts=7`)
- **Service Version:** `v1`

---

## 6. Independent Adversarial Red-Team Audit (Live RK3588J Board)
Raw log: [`results/adversarial_redteam_audit.json`](results/adversarial_redteam_audit.json)

- **Live-Wire Attacks Fired (IP-Only, Zero EdgeShield Port Touched):** `10` attack families (`468` raw packets) · Board Live-Monitor Events Delta: `+114`
- **Adversarial API Suite (`160` cases across Network, Phishing URL, Scam SMS, & Benign Controls):**
  - Attack Detection Rate (Recall): **100.0%** (`129/129`)
  - Benign False-Positive Rate: **12.9%** (`4/31`)
