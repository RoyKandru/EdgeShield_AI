# EdgeShield AI

### On-device threat, phishing and scam detection for Arm edge systems

**EdgeProc AI Technologies · Bharat AI-SoC Challenge 2026--27 · Problem
Statement 3**

EdgeShield AI is an on-device cybersecurity platform designed for
Arm-based edge systems. Its current target platform is the **Rockchip
RK3588J**, using three specialist Rockchip NPU inference workers for
network anomaly detection, phishing URL detection and SMS scam
detection, alongside an Arm64 CPU-based advisory pipeline using **Meta
Llama 3.2 1B Instruct through ExecuTorch**.

The design goal is to keep detection, event correlation, advisory
generation, local policy management and audit records on the edge
device, without requiring a cloud service for the core workflow.

> **Important distribution note:** The Llama `.pte` model and matching
> tokenizer are intentionally not included in this repository archive
> because of their size and separate model-distribution requirements.
> The EdgeShield integration and runner are included. The project team
> has separately tested offline ExecuTorch Llama inference on the
> RK3588J board. See [Llama model
> setup](#llama-32-1b-instruct-executorch-optional-separately-distributed-model).

------------------------------------------------------------------------

## Contents

-   [Project at a glance](#project-at-a-glance)
-   [Problem and approach](#problem-and-approach)
-   [Architecture](#architecture)
-   [Target hardware and software](#target-hardware-and-software)
-   [Detection pipelines](#detection-pipelines)
-   [Llama advisory pipeline](#llama-advisory-pipeline)
-   [Self-Correcting Intelligence](#self-correcting-intelligence-sci)
-   [Network monitoring and mobile SMS
    companion](#network-monitoring-and-mobile-sms-companion)
-   [Containment and safety](#containment-and-safety)
-   [Reported benchmark results](#reported-benchmark-results)
-   [Repository structure](#repository-structure)
-   [Installation and quick start](#installation-and-quick-start)
-   [Llama model
    setup](#llama-32-1b-instruct-executorch-optional-separately-distributed-model)
-   [API examples](#api-examples)
-   [Configuration](#configuration)
-   [Testing and benchmarking](#testing-and-benchmarking)
-   [Security, privacy and
    limitations](#security-privacy-and-limitations)
-   [Reproducibility and evidence](#reproducibility-and-evidence)
-   [Roadmap](#roadmap)
-   [Team and acknowledgements](#team-and-acknowledgements)
-   [License](#license)

------------------------------------------------------------------------

## Project at a glance

  ---------------------------------------------------------------------
  Item                               Description
  ---------------------------------- ----------------------------------
  Project                            EdgeShield AI

  Developer                          EdgeProc AI Technologies

  Challenge track                    Bharat AI-SoC Challenge 2026--27

  Problem statement                  PS3 --- On-device threat, phishing
                                     and scam detection

  Target SoC                         Rockchip RK3588J

  AI acceleration                    Three-core Rockchip NPU using RKNN

  Advisory model                     Llama 3.2 1B Instruct, via
                                     ExecuTorch

  Advisory execution                 Arm64 CPU; XNNPACK / Arm KleidiAI
                                     build as documented in the
                                     technical report

  Core detection channels            Network, URL and SMS

  Operating approach                 Local-first / offline-capable

  Main application                   Python, FastAPI and local web UI

  Native components                  C/C++ packet sniffer and
                                     ExecuTorch runner

  Local persistence                  SQLite

  Model distribution                 RKNN models included; Llama
                                     weights/tokenizer distributed
                                     separately
  ---------------------------------------------------------------------

### What EdgeShield does

1.  Analyses network-flow features and selected observable network
    metadata for suspicious activity.
2.  Analyses URLs for phishing characteristics using lexical and
    character-level features.
3.  Analyses submitted or companion-forwarded SMS text for scam and
    smishing characteristics.
4.  Produces a structured threat event with a verdict, risk information
    and supporting evidence.
5.  Uses a local Llama advisory pipeline, when its model files are
    installed, to explain a detector verdict and suggest safe next
    steps.
6.  Records feedback and evidence for a governed Self-Correcting
    Intelligence (SCI) workflow.
7.  Provides optional, controlled network containment actions with
    dry-run defaults and undo mechanisms.

**Design boundary:** specialist detectors produce the threat verdict.
Llama is an explanatory assistant; it is not the authority for changing
detector verdicts, security policy or containment state.

------------------------------------------------------------------------

## Problem and approach

Phishing, scam messages and network attacks often arrive through
different channels and are evaluated by separate tools. Cloud-dependent
services can be unsuitable for deployments where connectivity, privacy,
latency or data sovereignty matter.

EdgeShield brings three detection channels onto one Arm edge platform:

-   **Network:** flow-level anomaly and intrusion detection.
-   **URL:** phishing and suspicious-link detection.
-   **SMS:** smishing, impersonation and scam-message detection.

The system combines specialist models with shared event handling, local
advisory generation, policy governance and a local interface. Its
architecture is modular so that detection models can be updated without
making the LLM responsible for security decisions.

------------------------------------------------------------------------

## Architecture

EdgeShield uses two inference pipelines and a shared platform layer. The
SCI capability is under development; the complete closed-loop workflow
is a planned final-prototype milestone.

``` text
                  EDGE INPUTS
       ┌────────────┬────────────┬────────────┐
       │ Network    │ URL        │ SMS        │
       │ flows      │ links      │ text       │
       └─────┬──────┴─────┬──────┴─────┬──────┘
             │            │            │
             ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Network  │ │ URL      │ │ SMS      │
       │ NPU Core0│ │ NPU Core1│ │ NPU Core2│
       └─────┬────┘ └─────┬────┘ └─────┬────┘
             └────────────┼────────────┘
                          ▼
                ┌──────────────────┐
                │ ThreatEvent and  │
                │ risk/event layer │
                └────────┬─────────┘
                         │
             ┌───────────┴────────────┐
             ▼                        ▼
   ┌──────────────────┐     ┌────────────────────┐
   │ Local event store│     │ Advisory pipeline  │
   │ SQLite / audit   │     │ ExecuTorch + Llama │
   └────────┬─────────┘     └──────────┬─────────┘
            └────────────┬─────────────┘
                         ▼
             ┌────────────────────────┐
             │ Local UI, monitoring,  │
             │ event store and audit  │
             │ SCI foundation         │
             │ (under development)    │
             └────────────────────────┘
```

### Pipeline A --- specialist threat detection

Three persistent RKNN worker processes are configured for the three
RK3588 NPU cores:

  -----------------------------------------------------------------------
  NPU core                Detection task          Model family described
                                                  by project
  ----------------------- ----------------------- -----------------------
  Core 0                  Network IDS / anomaly   1D-CNN plus MLP
                          detection               

  Core 1                  Phishing URL detection  Lexical feature MLP
                                                  plus character 1D-CNN

  Core 2                  SMS smishing detection  Split BiGRU with masked
                                                  pooling
  -----------------------------------------------------------------------

The `.rknn` model files and associated metadata are located in
`models/rknn/`. The application also contains classical ML and heuristic
fallback paths. Actual NPU use depends on a compatible RKNN Lite
runtime, supported board/driver combination and successful worker
initialisation.

### Pipeline B --- local threat advisory

The advisory pipeline is separate from the detector workers:

-   Native ExecuTorch runner (`bin/executorch_runner`).
-   Llama 3.2 1B Instruct `.pte` model supplied separately.
-   Matching tokenizer supplied separately.
-   Local inference on the Arm64 CPU.
-   Prompt sanitisation and output validation in the application layer.
-   Deterministic template advisory fallback when model files or runtime
    are unavailable, subject to configuration.

The Llama pipeline explains structured detector results. It does not
replace the specialist classifiers.

### Shared platform services

-   FastAPI service and local dashboard.
-   Threat event and risk handling.
-   Local SQLite persistence.
-   SCI verification, diagnosis, policy proposal and validation
    components.
-   Platform monitoring and benchmark scripts.
-   Optional containment integrations.

------------------------------------------------------------------------

## Target hardware and software

### Target board

The project technical report identifies the target as a **Rockchip
RK3588J Arm64 board with a three-core NPU**.

The report records the following software/runtime combination:

  ---------------------------------------------------------------------
  Component                          Reported environment
  ---------------------------------- ----------------------------------
  SoC                                Rockchip RK3588J

  CPU architecture                   Arm64 / AArch64

  NPU runtime                        RKNN Toolkit Lite2 2.3.2

  Rockchip runtime/driver            Report references RKNPU driver
                                     0.9.6 and librknnrt 2.3.x

  Advisory runtime                   ExecuTorch

  Advisory backend                   XNNPACK with Arm KleidiAI, as
                                     reported

  Application                        Python 3.10+ / FastAPI

  Local database                     SQLite
  ---------------------------------------------------------------------

Versions can vary by board image and deployment. Confirm the versions
installed on the target before reproducing the measurements.

### Host development

Model training and conversion scripts are provided for development
workflows. Training datasets and the large Llama model are not bundled.
The pre-converted RKNN model artifacts are included in this source
archive.

------------------------------------------------------------------------

## Detection pipelines

### 1. Network intrusion and anomaly detection

The network model consumes a 26-feature flow representation produced by
the project feature pipeline. The technical report describes a compact
1D-CNN and MLP model.

The live monitor can observe selected packet and flow metadata on a
configured network interface. It aggregates flow information and sends
suitable features to the network detector.

**Scope:** this is not a TLS decryption product. Encrypted application
payloads are not readable through passive packet capture.

### 2. Phishing URL detection

The URL model combines lexical URL features with character-level
features. It is intended to detect suspicious URL structure, domain
patterns and phishing indicators.

The technical report describes a 25-feature lexical branch and a
character CNN branch. The exact preprocessing and feature ordering used
by the model must remain consistent between training, conversion and
inference.

### 3. SMS scam and smishing detection

The SMS detector analyses message text and associated available
metadata. The report describes a BiGRU-based model and
training/evaluation using public SMS datasets plus an Indian scam/OTP
dataset assembled for the project.

SMS input can be submitted through the API or delivered by the optional
Android/Termux companion. The gateway does **not** obtain arbitrary
handset SMS content from passive Ethernet packet capture.

### Detector metrics

The technical report contains held-out detector results. Metrics must be
interpreted with their operating thresholds; results at different
thresholds must not be combined into one row.

  -----------------------------------------------------------------------
  Detector                Reported result         Threshold context
  ----------------------- ----------------------- -----------------------
  Network IDS             ROC-AUC 1.0000; F1      Held-out evaluation in
                          0.9985; precision       the technical report
                          0.9999; recall 0.9971;  
                          FPR 0.00%               

  URL                     ROC-AUC 0.9781; average Precision/FPR operating
                          precision 0.9780;       point at threshold
                          precision 0.9878; FPR   0.96; recall is
                          0.80%                   threshold-dependent

  SMS                     ROC-AUC 0.9942; F1      F1 reported at
                          0.9592                  threshold 0.50; the
                                                  report separately
                                                  reports recall 0.9793
                                                  at threshold 0.22
  -----------------------------------------------------------------------

These are dataset evaluation results, not a guarantee of real-world
detection performance. Dataset composition, split strategy, threshold
and false-positive rate matter. See the detailed benchmark JSON and
technical report before reusing the figures.

------------------------------------------------------------------------

## Llama advisory pipeline

### Llama 3.2 1B Instruct via ExecuTorch

The project team reports having tested offline ExecuTorch Llama
inference on the physical RK3588J board. The `.pte` weights and matching
tokenizer are intentionally excluded from the source ZIP because of
their size.

The repository includes the integration code and native runner, but
**does not include the model weights or tokenizer**. A clean clone
therefore cannot reproduce Llama generation until the separately
distributed model files are installed.

### Model package expected by the application

The application README/configuration refers to these default paths:

``` text
models/llama.pte
models/tokenizer.model
```

The top-level project README also references an exported artifact named
`llama_3_2_1b_xnnpack.pte` and `tokenizer.json`. Use the exact filenames
and tokenizer format that match the runner build you validated on the
board. Do not rename or mix incompatible model/tokenizer exports.

### Setup

1.  Obtain the authorised/exported Llama 3.2 1B Instruct ExecuTorch
    artifact and matching tokenizer from the project team's model
    package.
2.  Verify the artifact checksum supplied by the team.
3.  Copy the model and tokenizer to the configured paths on the board.
4.  Ensure the ExecuTorch runner binary is built for the board's AArch64
    environment and matches the model export/runtime.
5.  Configure the paths using the environment variables below.
6.  Start EdgeShield and verify the advisory provider reports the
    ExecuTorch backend rather than the template fallback.
7.  Run a known test prompt with network access disabled to verify
    offline operation.

Example environment configuration (adjust paths and tokenizer filename
to match your validated package):

``` bash
export EDGESHIELD_EXECUTOR=command
export EDGESHIELD_EXECUTOR_COMMAND="$PWD/bin/executorch_runner"
export EDGESHIELD_EXECUTOR_MODEL_PATH="$PWD/models/llama.pte"
export EDGESHIELD_EXECUTOR_TOKENIZER_PATH="$PWD/models/tokenizer.model"
export EDGESHIELD_EXECUTOR_DAEMON=true
```

**Reproducibility requirement:** publish the exact model identifier,
export recipe, artifact size, SHA-256, tokenizer identity, ExecuTorch
build/runtime version and board-side test log. Do not commit the large
model file or credentials to Git.

### Reported Llama benchmark

The technical report and benchmark JSON record:

-   20-run sustained test.
-   21.27 tokens/second reported.
-   p50 latency 1,885.5 ms.
-   Peak recorded CPU temperature 88.6°C.
-   No throttling reported during that test.

These are project-recorded board measurements. The 20-run test does not
by itself establish long-duration thermal stability under all detector,
network-monitor and advisory workloads running together.

------------------------------------------------------------------------

## Self-Correcting Intelligence (SCI) --- planned for the final prototype

SCI is a planned capability intended to evolve EdgeShield from a
detection-and-advisory system into a governed, evidence-grounded
self-correction platform.

**Implementation status:** SCI is not fully implemented in the current
submission source. The repository contains initial SCI-related modules,
policy data structures and tests, but the complete nine-stage
closed-loop workflow has not yet been delivered or validated end to end.
The full SCI implementation is planned for the final prototype.

The intended workflow is:

1.  Detection and event recording.
2.  Feedback and outcome collection.
3.  Independent evidence verification.
4.  Failure and root-cause diagnosis.
5.  Scoped correction or policy proposal.
6.  Golden-suite regression validation.
7.  Controlled policy deployment.
8.  Rollback and recovery.
9.  Audit and continuous monitoring.

These stages describe the target design, not a claim that all stages are
currently operational.

### Intended safety principles

-   User feedback must not automatically be treated as ground truth.
-   Corrections should require independently verifiable evidence.
-   Uncertain outcomes should remain uncertain rather than being forced
    into malicious/benign labels.
-   Policy changes should be scoped and validated before deployment.
-   Failed validation should block deployment.
-   Changes should be auditable and reversible.
-   Llama-generated explanations must not independently deploy policies
    or alter detector verdicts.

### Current source status

Initial SCI-related code is located under `app/sci/`, with policy data
under `data/sci_policy/` and related tests under `tests/`. These
components represent groundwork toward the planned SCI architecture.
They must not be interpreted as proof of a completed autonomous
learning, correction, deployment or rollback loop.

For this challenge submission, SCI should be presented as **in
development / planned for the final prototype**. Any existing SCI
unit-test or ledger result should be described narrowly as a test of
that specific component, not as end-to-end SCI validation.

------------------------------------------------------------------------

## Network monitoring and mobile SMS companion

### Passive network monitoring

The native `bin/live_sniffer` component captures packets from a selected
Linux network interface. The Python live monitor reduces observed
traffic into flow features and extracts selected information from
protocols where content is visible.

Capabilities are constrained by what the interface can observe and what
is exposed in plaintext.

### Android / Termux SMS companion

The project includes `scripts/android_sms_forwarder.py` and
`docs/MOBILE_COMPANION.md` for forwarding SMS data from a companion
device to the EdgeShield gateway over a local network.

This is an explicit companion integration. It is not equivalent to
unrestricted SMS access by the gateway. Users must grant the required
Android permissions and configure the forwarding endpoint themselves.

------------------------------------------------------------------------

## Containment and safety

The source includes an optional containment ladder with support
described for:

-   Domain sinkholing through dnsmasq.
-   IP blocking through nftables.
-   MAC quarantine through nftables.
-   Dry-run mode by default.
-   Time-limited actions and undo endpoints.
-   Protected local/gateway/DNS hosts.

**Operational warning:** containment changes can interrupt network
connectivity. Keep dry-run enabled during initial setup. Test actual
enforcement and expiry/undo on an isolated lab network before enabling
it on a production network. Review the current implementation and system
configuration; do not assume that an application-side expiry record
automatically removes every operating-system rule or DNS configuration
file.

The link sandbox is optional and should remain disabled unless its
network isolation and SSRF protections have been reviewed for the
deployment environment.

------------------------------------------------------------------------

## Reported benchmark results

The following results are transcribed from the project's supplied
technical report and benchmark JSON files. They are **reported project
measurements**, not independently reproduced by this README.

  ---------------------------------------------------------------------
  Benchmark                        Reported result Interpretation
  -------------------- --------------------------- --------------------
  NPU Core 0 ---                 128.89 requests/s Individual worker
  Network                                          throughput

  NPU Core 1 --- URL             124.89 requests/s Individual worker
                                                   throughput

  NPU Core 2 --- SMS              61.75 requests/s Individual worker
                                                   throughput

  Three-core combined            315.53 requests/s Concurrent NPU
                                                   benchmark

  Llama sustained                   21.27 tokens/s 20-run test
  generation                                       

  Llama p50 latency                     1,885.5 ms 20-run test

  Llama peak CPU                            88.6°C Recorded in the same
  temperature                                      test

  Pipeline A memory                      485.02 MB NPU workers, API and
                                                   sniffer in report

  Pipeline B memory                    1,681.07 MB Llama daemon in
                                                   report

  Total recorded                       2,166.09 MB Sum reported in the
  process RSS                                      technical report

  Red-team attacks                         129/129 Reported API
  detected                                         adversarial suite

  Red-team benign                             4/31 12.9% FPR in the
  false positives                                  reported benign
                                                   controls

  Automated regression        61 passed, 3 skipped Reported test run;
                                                   reproduce on target
  ---------------------------------------------------------------------

### Important interpretation

-   The three-core throughput is a concurrent benchmark result; it
    should not be interpreted as end-to-end user throughput for every
    mixed workload.
-   The Llama test is separate from the NPU concurrency test.
-   The red-team suite reports 100% attack recall but also a 12.9%
    false-positive rate on its benign controls. Both results must be
    disclosed together.
-   Detector model metrics, adversarial API metrics and live-wire
    monitoring counts use different test populations and should not be
    merged.
-   Temperature and memory measurements depend on board cooling, OS
    image, background processes and workload.

Raw proof files are in `benchmarks/results/`.

------------------------------------------------------------------------

## Repository structure

``` text
source/edgeshield-ai/
├── app/
│   ├── main.py                 # FastAPI application and routes
│   ├── auth.py                 # API authentication helpers
│   ├── config.py               # Runtime configuration
│   ├── events.py               # Threat event handling
│   ├── risk.py                 # Risk logic
│   ├── advisory/                # ExecuTorch adapter and template fallback
│   ├── detectors/               # Network, URL, SMS and live monitoring
│   ├── edge_models/             # RKNN clients and worker processes
│   ├── ml/                      # Feature extraction and ML runtime helpers
│   ├── sci/                     # Initial SCI groundwork; full workflow planned
│   ├── platform/                # System monitoring
│   └── static/                  # Local dashboard
├── bin/
│   ├── live_sniffer             # Native packet capture binary
│   └── executorch_runner        # Native ExecuTorch runner binary
├── src/                         # Native C/C++ sources
├── models/
│   ├── rknn/                    # Three RKNN models and metadata
│   └── *.joblib                 # Classical ML model artifacts
├── benchmarks/
│   └── results/                 # Benchmark and evaluation JSON
├── scripts/
│   ├── edge_models/             # Training, conversion and benchmark scripts
│   ├── run_benchmarks.py
│   ├── export_llama_executorch.sh
│   └── android_sms_forwarder.py
├── deploy/                      # systemd and nftables examples
├── docs/                        # Architecture, deployment, demo and SCI docs
├── tests/                       # Python tests
├── native/                      # Optional standalone C++20 baseline
├── data/                        # Local policy data; runtime DB is created locally
├── pyproject.toml
└── run.sh
```

------------------------------------------------------------------------

## Installation and quick start

### 1. Clone and enter the application directory

``` bash
git clone <PUBLIC_REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>/source/edgeshield-ai
```

Replace the placeholders with the actual public repository URL and
directory name after publication.

### 2. Create a Python environment

Python 3.10 or newer is required by `pyproject.toml`.

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Configure the RKNN Lite worker environment

On the RK3588J board, install a compatible `rknn-toolkit-lite2` package
in the Python environment used by the worker processes. The project
documentation records testing with RKNN Lite2 2.3.2.

Example:

``` bash
python3.11 -m venv .venv-rknn
source .venv-rknn/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy rknn-toolkit-lite2==2.3.2
```

Return to the main application environment and set the worker
interpreter path:

``` bash
export EDGESHIELD_RKNN_PYTHON="$PWD/.venv-rknn/bin/python"
```

Use the actual interpreter path on your board. The RKNN runtime, NPU
driver and model conversion versions must be compatible.

### 4. Grant packet-capture capabilities (optional)

The passive sniffer may require Linux capabilities. Only grant these to
the reviewed binary and only if live monitoring is required:

``` bash
sudo setcap cap_net_raw,cap_net_admin=eip bin/live_sniffer
getcap bin/live_sniffer
```

If you do not need passive monitoring, disable it using the
configuration variable below.

### 5. Configure authentication and runtime

Do not expose the API on a shared network with authentication disabled.
Generate a strong random token and configure it before starting:

``` bash
export EDGESHIELD_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

For a local-only test, bind to loopback:

``` bash
export EDGESHIELD_HOST=127.0.0.1
export EDGESHIELD_PORT=8000
```

### 6. Start EdgeShield

``` bash
./run.sh
```

The project launcher also documents these commands:

``` bash
./run.sh status
./run.sh logs
./run.sh restart
./run.sh stop
```

Check the launcher and environment configuration on the target board
before installing it as a service.

### 7. Open the local interface

``` text
http://127.0.0.1:8000
```

If the service is bound to a different interface, use the corresponding
address from a trusted local network. Do not expose the service directly
to the public internet.

------------------------------------------------------------------------

## Llama 3.2 1B Instruct (ExecuTorch): optional separately distributed model

The large Llama model and tokenizer are excluded from this source
repository. The team has separately tested the model running offline on
the RK3588J board.

Before distributing a model package, provide:

-   Exact model identifier and licence/source.
-   Exact `.pte` filename and file size.
-   Matching tokenizer filename and format.
-   SHA-256 checksums for both artifacts.
-   ExecuTorch version and build configuration.
-   Export command or reproducible export script.
-   Board-side launch command and successful offline inference log.

Do not download an arbitrary `.pte` or tokenizer and assume it is
compatible with the included runner.

The application defaults documented in the source are:

``` bash
EDGESHIELD_EXECUTOR=command
EDGESHIELD_EXECUTOR_COMMAND=./bin/executorch_runner
EDGESHIELD_EXECUTOR_MODEL_PATH=./models/llama.pte
EDGESHIELD_EXECUTOR_TOKENIZER_PATH=./models/tokenizer.model
EDGESHIELD_EXECUTOR_DAEMON=true
```

If the model files are absent, the application can use the deterministic
template advisory provider. In that mode, do not claim that Llama
inference is active.

------------------------------------------------------------------------

## API examples

The examples below assume the API is running on the local machine. The
exact request schemas are defined in the source application.

### Analyse a URL

``` bash
curl -sS -X POST "http://127.0.0.1:8000/api/url/analyze" \
  -H "Content-Type: application/json" \
  -H "X-EdgeShield-Token: $EDGESHIELD_API_TOKEN" \
  -d '{
    "url": "https://example.invalid/verify"
  }'
```

Use a safe test URL or a controlled lab URL when demonstrating
detection. Do not visit suspected malicious URLs on a production device.

### Analyse an SMS message

``` bash
curl -sS -X POST "http://127.0.0.1:8000/api/sms/analyze" \
  -H "Content-Type: application/json" \
  -H "X-EdgeShield-Token: $EDGESHIELD_API_TOKEN" \
  -d '{
    "sender": "VM-ALERT",
    "message": "Your account requires an urgent verification. Do not share OTPs or passwords."
  }'
```

### Analyse network flow features

``` bash
curl -sS -X POST "http://127.0.0.1:8000/api/network/analyze" \
  -H "Content-Type: application/json" \
  -H "X-EdgeShield-Token: $EDGESHIELD_API_TOKEN" \
  -d '{
    "flows": [{
      "src_ip": "192.168.1.50",
      "dst_ip": "192.168.1.6",
      "dst_port": 22,
      "protocol": "tcp",
      "packets": 60,
      "bytes": 5400,
      "syn_packets": 55,
      "failed_connections": 24,
      "duration_seconds": 1.5
    }]
  }'
```

### Check monitoring and inspect the current SCI foundation

``` bash
curl -sS "http://127.0.0.1:8000/api/network/monitor/status"
curl -sS "http://127.0.0.1:8000/api/sci/pipeline/status"
```

If the deployment uses a different authentication policy, follow the
configured API security requirements. The examples should be checked
against the current API schema before publication.

------------------------------------------------------------------------

## Configuration

The following variables are documented by the source. Confirm actual
defaults in `app/config.py` before deployment.

  -----------------------------------------------------------------------
  Variable                               Purpose
  -------------------------------------- --------------------------------
  `EDGESHIELD_RKNN_PYTHON`               Python interpreter containing
                                         RKNN Lite2

  `EDGESHIELD_EXECUTOR`                  Advisory provider selection
                                         (`command` or `template`)

  `EDGESHIELD_EXECUTOR_COMMAND`          Path to native ExecuTorch runner

  `EDGESHIELD_EXECUTOR_MODEL_PATH`       Llama `.pte` model path

  `EDGESHIELD_EXECUTOR_TOKENIZER_PATH`   Matching tokenizer path

  `EDGESHIELD_EXECUTOR_DAEMON`           Keep advisory model runner
                                         resident

  `EDGESHIELD_LIVE_MONITOR_ENABLED`      Enable or disable passive live
                                         monitoring

  `EDGESHIELD_LIVE_MONITOR_INTERFACE`    Network interface to monitor

  `EDGESHIELD_LIVE_MONITOR_WINDOW_S`     Flow aggregation window

  `EDGESHIELD_DB_PATH`                   Local SQLite database path

  `EDGESHIELD_API_TOKEN`                 API authentication token

  `EDGESHIELD_HOST`                      API bind address

  `EDGESHIELD_PORT`                      API port

  `EDGESHIELD_CONTAINMENT_DRY_RUN`       Keep containment in dry-run mode
                                         by default

  `EDGESHIELD_SANDBOX_ENABLED`           Enable optional URL link sandbox
  -----------------------------------------------------------------------

For a safer local-only baseline:

``` bash
export EDGESHIELD_HOST=127.0.0.1
export EDGESHIELD_API_TOKEN="<GENERATE_A_LONG_RANDOM_SECRET>"
export EDGESHIELD_LIVE_MONITOR_ENABLED=false
export EDGESHIELD_CONTAINMENT_DRY_RUN=true
export EDGESHIELD_SANDBOX_ENABLED=false
```

------------------------------------------------------------------------

## Testing and benchmarking

### Run the portable test suite

``` bash
pytest -q
```

Some tests require RKNN Lite2 and physical RK3588 hardware. A test
skipped because hardware is unavailable is not a successful hardware
validation.

### Run the benchmark scripts

``` bash
python scripts/run_benchmarks.py
python scripts/edge_models/benchmark_concurrency.py
```

Check each script's help/options and required runtime dependencies
before execution. Store new results with the board model, OS image,
kernel, NPU driver, RKNN runtime, Python version, model hashes, test
command and timestamp.

### Recommended target-board validation

For a reproducible release, capture:

-   Each NPU worker's initialisation log and assigned core.
-   Per-detector inference latency and throughput.
-   Concurrent three-worker throughput and error rate.
-   CPU, NPU and memory utilisation.
-   Temperature during sustained mixed workloads.
-   Llama model/tokenizer checksums and offline inference log.
-   API end-to-end latency.
-   Evidence of current SCI component tests only; do not claim
    end-to-end SCI validation until the final workflow is implemented.
-   Test-suite summary, including skipped tests and why they were
    skipped.

Do not report a benchmark as a board result unless it was collected on
the target board. Do not report a test as passed when it was skipped.

------------------------------------------------------------------------

## Security, privacy and limitations

### Privacy and offline operation

The core design is local-first. Network traffic, URLs, SMS inputs,
threat events and model inference can be processed on the edge device.
The project does not require a cloud API for the core detection and
advisory workflow.

The optional mobile companion forwards SMS data to the local gateway.
This is a separate, user-configured integration.

### Known technical limitations

1.  **TLS, DoH and encrypted traffic:** passive packet capture cannot
    decrypt encrypted application payloads. It can inspect flow metadata
    and supported plaintext protocol fields.
2.  **Cellular SMS:** handset SMS is supplied through the companion
    forwarding workflow; it is not read from arbitrary phones by the
    passive network sniffer.
3.  **Compromised legitimate domains:** lexical URL detection may not
    identify every malicious page hosted on a legitimate domain.
    Optional page inspection has separate network-security implications.
4.  **False positives:** the supplied red-team evaluation reports 4
    false positives among 31 benign controls (12.9%). This must be
    improved and re-evaluated on a larger, independent benign dataset.
5.  **Thermal behaviour:** the supplied 20-run Llama test records a peak
    temperature of 88.6°C. Longer mixed-workload thermal validation is
    still important.
6.  **Model dependency:** the Llama weights and tokenizer are not
    included. Llama mode requires separately supplied compatible files.
7.  **Dataset generalisation:** public and project datasets do not
    represent every real-world threat. Results depend on data quality,
    split design and operating thresholds.
8.  **Containment risk:** firewall and DNS changes can disrupt
    connectivity. Keep dry-run enabled until enforcement, expiry and
    rollback have been tested in an isolated environment.
9.  **Hardware dependence:** RKNN inference requires a compatible
    Rockchip board, driver, runtime and model. Other systems may use
    CPU/classical fallback paths and will not reproduce RK3588 NPU
    performance.

### Responsible-use guidance

-   Use only on networks and devices you own or are authorised to
    monitor.
-   Obtain consent and comply with local laws before collecting or
    forwarding SMS or network data.
-   Do not use containment actions on production networks without
    explicit operator approval and a tested recovery path.
-   Treat model outputs as security signals requiring context, not as
    infallible determinations.

------------------------------------------------------------------------

## Reproducibility and evidence

The repository includes benchmark evidence under `benchmarks/results/`
and technical documentation under `docs/`.

Primary documents:

-   `docs/EdgeShield_AI_Technical_Report_v1.4.pdf`
-   `docs/TECHNICAL_REPORT_v1.4.md`
-   `docs/BOARD_DEPLOYMENT_RK3588J.md`
-   `docs/DEMO.md`
-   `docs/SCI_ENGINE_DESIGN.md`
-   `docs/MOBILE_COMPANION.md`
-   `benchmarks/README.md`

Before final submission, ensure that:

-   The public repository contains the final source and report.
-   The report version matches the software release identifier.
-   Benchmark JSON files correspond to the exact source/model revision.
-   Llama model metadata and checksums are published even if the model
    binary is not.
-   The public demo video URL is added below.
-   Screenshots and demo files referenced by this README actually exist
    in the repository.
-   No secrets, tokens, private data, raw personal SMS or restricted
    datasets are committed.

### Submission links

-   **Technical report:**
    [`docs/EdgeShield_AI_Technical_Report_v1.4.pdf`](docs/EdgeShield_AI_Technical_Report_v1.4.pdf)
-   **Demo video:** *Add the public two-minute demo URL here.*
-   **Challenge submission:** Bharat AI-SoC Challenge 2026--27, PS3.

------------------------------------------------------------------------

## Roadmap

The next engineering priorities are:

1.  Harden API authentication and privileged containment execution.
2.  Ensure domain containment has real operating-system-level expiry and
    reliable undo.
3.  Complete the planned SCI workflow, including independent evidence
    verification, diagnosis, gated correction, deployment, rollback and
    audit.
4.  Improve benign-case false-positive performance and report
    per-threshold metrics consistently.
5.  Validate model preprocessing and inference parity across training,
    conversion and target deployment.
6.  Run long-duration mixed NPU, network-monitor and Llama workloads
    with thermal and memory logging.
7.  Improve clean-clone reproducibility, version metadata and model
    artifact provenance.

------------------------------------------------------------------------

## Team and acknowledgements

**EdgeProc AI Technologies**

Project: **EdgeShield AI**

Challenge: **Bharat AI-SoC Challenge 2026--27 --- Problem Statement 3**

Add the registered team name, team members, mentor and institutional
details here before final submission.

The project uses and builds upon open-source software, public research
datasets and the Rockchip/ExecuTorch ecosystems. See the corresponding
upstream licences and dataset terms before redistribution.

------------------------------------------------------------------------

## License

A repository-level licence has not been asserted in this README. Add the
licence approved by the project owner before making the repository
public. Until then, do not assume that the source, model artifacts,
datasets or third-party components are licensed for unrestricted reuse.

Third-party models, datasets, binaries and libraries may have separate
licence terms. Those terms continue to apply independently of the
EdgeShield source repository.
