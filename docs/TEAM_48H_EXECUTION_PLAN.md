# EdgeShield AI — Team Execution Plan for the 25th Submission

**Purpose:** A practical, tightly scoped execution brief for the final two working days before the initial submission deadline. The deadline is stated by the project owner as the 25th. Confirm the official portal cut-off time and timezone immediately.

## 1. Mission and product framing
EdgeShield AI is a long-term EdgeProc product initiative. The Bharat AI SoC Challenge is the first validation milestone, not a throwaway hackathon application. The product direction is a reusable EdgeProc C++ Edge Intelligence Framework that can later serve other EdgeProc products.

The 2nd-edition challenge emphasis is SoC product engineering: algorithm development, optimisation, system integration, edge deployment, benchmarking, validation and product engineering. Our submission must therefore show the algorithm-to-SoC engineering path, not only a dashboard or an AI application.

## 2. Freeze the submission scope
**Exactly three external inputs:**
- Network: offline PCAP/PCAPNG analysis.
- URL: explicit URL analysis; never fetch/open the submitted URL during classification.
- SMS: explicit message analysis.

**Internal SoC telemetry (not extra inputs):** CPU, RAM, storage, temperature, CPU frequency, NPU/GPU availability/utilisation only when the kernel exposes real counters, and inference/runtime health.

**Required challenge path:** Llama-family advisory model on-device through ExecuTorch. Optional RKNN acceleration is a separate path for compatible detector models; it does not replace ExecuTorch.

## 3. Architecture decision: C++ first, incremental migration
- C++20/CMake is the target production edge runtime for detectors, event contracts, risk fusion, SCI, telemetry, hardware abstraction and accelerator adapters.
- Python remains for training, dataset preparation, experiments and evaluation.
- Existing Python/FastAPI/SQLite/dashboard prototype remains the demonstrator/compatibility surface until native replacements pass parity tests.
- Do not attempt a complete rewrite in two days. Build and demonstrate a coherent, testable native core foundation, while retaining the working three-pipeline prototype.

## 4. Current baseline — what exists and what does not
### Existing prototype (v1.1.0)
- Python URL and SMS heuristic detectors.
- PCAP/PCAPNG network flow analysis using Scapy and heuristic rules.
- Unified event/risk model, FastAPI routes, SQLite persistence and dashboard.
- SCI incident recording, rule-based diagnosis and allow-listed corrective recommendations.
- Adaptive feedback/candidate-learning scaffolding.
- System Monitor tab backed by Linux `/proc`, `/sys` and filesystem metrics where available.
- ExecuTorch command adapter scaffold.

### New native foundation (v1.2.0)
- C++20/CMake build.
- Shared ThreatEvent/evidence types and detector interface.
- URL/SMS lexical heuristic examples.
- Starter risk-fusion component.
- Linux telemetry sampler for memory, root filesystem and thermal zone where exposed.
- Native smoke tests.

### Not yet demonstrated / do not claim as complete
- Full C++ port of the three pipelines or SCI.
- Native C++ network/PCAP detector.
- Native monitor wired to the dashboard.
- Actual Llama `.pte` + tokenizer + ExecuTorch C++ inference on RK3588J.
- RKNN C++ inference for a threat detector. The existing YOLO RKNN model is a vision model and is not a URL/SMS/network classifier.
- Dataset-validated accuracy, calibrated confidence, power benchmark, thermal qualification or long-duration soak test.

## 5. Team workstreams and ownership
Assign one named owner and one reviewer to each workstream at kickoff. Names are intentionally left blank for the team lead to allocate.

| Workstream | Owner | Deliverable by submission | Priority |
|---|---|---|---|
| A. Integration / release lead | ______ | Freeze scope, integrate branches, run release checklist, package source and evidence | P0 |
| B. Native C++ core | ______ | Build C++20 core on ARM64; run native smoke tests; document native/legacy boundary | P0 |
| C. Detection pipelines | ______ | Demonstrate URL, SMS and PCAP flows from existing app; prepare reproducible test inputs and outputs | P0 |
| D. SoC / ExecuTorch | ______ | Verify actual model/runtime path if available; otherwise produce precise blocker/evidence and avoid false readiness claims | P0 |
| E. System Monitor / hardware | ______ | Capture CPU/RAM/storage/temperature and NPU driver/runtime availability; report unsupported counters honestly | P1 |
| F. Benchmark / validation | ______ | Record per-pipeline latency, throughput where meaningful, RSS, CPU, environment and test corpus; produce results table | P0 |
| G. Submission / demo | ______ | Final narrative, architecture diagram, short demo, screenshots, README and portal submission | P0 |

Small team members may own more than one workstream, but every P0 deliverable must have one accountable owner.

## 6. Two-day execution schedule
### Day 1 — Stabilise, build and prove the vertical slice
1. Freeze scope and assign owners; do not add features.
2. Pull the v1.2.0 source package; build native C++ core on the Debian 12 ARM64 board using CMake.
3. Run CTest and native URL/SMS/monitor smoke commands; save terminal logs.
4. Start existing Python app and demonstrate all three existing pipelines: URL, SMS and PCAP/PCAPNG.
5. Verify System Monitor readings against Linux commands (`/proc/meminfo`, `df`, thermal sysfs); capture board model, OS, kernel and runtime versions.
6. Verify ExecuTorch path. If the Llama model, tokenizer, runner or runtime is not ready, document exact missing item. Do not spend the whole day on speculative model conversion.
7. Create a single reproducible evidence folder with commands, outputs, screenshots, input hashes and version metadata.

**Day 1 exit gate:** Native build passes; legacy application launches; all three input demos execute; hardware facts are captured; ExecuTorch status is evidence-backed.

### Day 2 — Integrate, benchmark, validate and submit
1. Fix only P0 integration and reproducibility defects.
2. Run repeatable tests for URL, SMS and PCAP inputs; record dataset/sample provenance and expected outcomes.
3. Measure per-request latency (warm-up + repeated runs), memory/RSS and CPU utilisation; record board temperature before/after. Do not publish numbers without test conditions.
4. Run offline demonstration with WAN disconnected if the local advisory/model path is genuinely available.
5. Verify UI/API, SCI incident creation and ledger verification; clearly label SCI actions as recommendations if no recovery executor exists.
6. Prepare architecture/HLD/SLD, implementation status, benchmark report, limitations and roadmap.
7. Rehearse a 3–5 minute demo; test the exact presentation device and submission files.
8. Submit ahead of portal cut-off; retain a timestamped copy and confirmation.

**Day 2 exit gate:** reproducible demo, source archive, architecture and evidence pack are final; all claims map to proof; submission receipt saved.

## 7. Board bring-up and commands
Known engineering target from prior board validation: EdgeBLE-6B Industrial, RK3588J, ARM64, Debian 12, Linux kernel 6.1.43-v2.02-g40fccbc31002. RKNN Runtime 2.3.0 and RKNPU driver 0.9.6 were detected; `/dev/dri/renderD129` exists and the `open` user was added to `render`. A YOLOv8n RK3588 `.rknn` model loaded and NPU runtime initialised successfully. This is runtime initialisation evidence only, not threat-model inference evidence.

Build native core:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j2
ctest --test-dir native/build --output-on-failure
```

Run native smoke tests:
```bash
native/build/edgeshield_native_demo url 'http://secure-login-example.com/verify'
native/build/edgeshield_native_demo sms 'Urgent: verify your account and share OTP'
native/build/edgeshield_native_demo monitor unused
```

Check board:
```bash
uname -a
cat /etc/os-release
free -h
df -h /
ls -l /dev/dri
ls -l /sys/class/thermal/thermal_zone*/temp 2>/dev/null
```

For the vendor RKNN Lite Python package, the validated interpreter was `/usr/bin/python3`; Miniforge Python 3.14 did not have `rknnlite`. Do not modify the board's vendor runtime installation during the submission sprint.

## 8. Benchmark protocol
For each detector/runtime, report:
- Board/SoC, OS, kernel, compiler, build flags and runtime/driver versions.
- Model and dataset/sample hashes; input sizes and number of runs.
- Warm-up count, measured iteration count, p50/p95 latency and throughput if applicable.
- Peak RSS, CPU utilisation, available NPU counter and thermal readings.
- Offline state and whether inference is CPU, NPU or ExecuTorch.
- Known limitations and whether the measurement is prototype-only or qualified.

Do not compare latency numbers across different input sizes, backends or thermal conditions without qualification. Never infer NPU utilisation from device-node presence.

## 9. Submission story
**Problem:** scam communications, phishing URLs and suspicious network behaviour require local, privacy-preserving detection, especially where cloud connectivity cannot be assumed.

**Solution:** EdgeShield AI combines three independently scoped detection pipelines, a common threat-event/risk layer, evidence-grounded local advisory, SCI and SoC resource monitoring.

**SoC engineering contribution:** C++-first runtime foundation, ARM64 build, CPU baseline, capability-aware accelerator strategy, mandatory ExecuTorch Llama advisory path, optional separate RKNN detector acceleration, telemetry and reproducible benchmarking.

**Long-term product:** a reusable EdgeProc Edge Intelligence Framework, with Arm Bharat AI SoC Challenge as the first validation milestone.

## 10. Final release checklist
- [ ] Official deadline time/timezone confirmed; submission owner assigned.
- [ ] Three inputs only: Network, URL, SMS.
- [ ] Native C++ build and CTest pass on target or compatible ARM64 environment.
- [ ] Existing application starts and all three pipelines demonstrated.
- [ ] System Monitor metrics checked against OS sources; unsupported counters shown as unavailable.
- [ ] ExecuTorch/Llama state labelled accurately; actual on-device inference evidence attached if completed.
- [ ] RKNN evidence not misrepresented as threat-classifier inference.
- [ ] Benchmark conditions and raw results retained.
- [ ] SCI recommendations not described as executed recovery.
- [ ] Offline claim tested, not assumed.
- [ ] Source, architecture, demo, benchmark and limitations included.
- [ ] Demo rehearsed; submission receipt saved.

## 11. Release language discipline
Use the terms **Implemented**, **Tested**, **Benchmarked**, **Qualified** and **Planned** precisely. A code path existing does not mean it has been tested on the SoC; a successful runtime initialisation does not mean a model inference has been benchmarked; and a benchmark does not equal product qualification.
