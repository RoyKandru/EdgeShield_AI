# EdgeShield AI  — Python-first detector + RK3588J execution plan

## Product and challenge architecture

This release keeps the v1.3.0 Item 2 Python detector implementation and its evaluation reports as the reference baseline. It adds a Python-first edge orchestration layer for three compact neural detector candidates and an RKNNLite adapter that requests one RK3588 NPU core per detector. It does not replace the Item 2 `.joblib` models or claim that the new models are trained, converted, or validated on the board.

### Pipeline A — specialist detection (NPU target)

| Workload | Candidate architecture | Assigned core | Input representation |
|---|---|---:|---|
| Network IDS | Quantisation-ready 1D-CNN | NPU core 0 | 26 reproducible `MODEL_FEATURES` from the existing `FlowReducer` |
| Phishing URL | Character-level 1D-CNN | NPU core 1 | Fixed-length one-hot character tensor (59 channels, 512 chars by default) |
| SMS/smishing | Character-level 1D-CNN | NPU core 2 | Fixed-length one-hot character tensor (57 channels, 512 chars by default) |

The SMS BiGRU is retained as a research/CPU candidate in `app/edge_models/models.py`. It is not the default NPU export because GRU/operator support and quantisation support depend on the installed RKNN Toolkit2 version. Prefer the character CNN for the first RKNN attempt; compare BiGRU only after operator compatibility is demonstrated.

### Pipeline B — Llama advisory (Arm CPU)

Llama-family inference is a separate path. The event produced by the specialist detector is passed to a device-specific ExecuTorch runner. The existing `app/advisory/executorch_command.py` is an adapter contract only; an actual Llama `.pte`, tokenizer, prompt template, generation loop and ExecuTorch runtime must be installed and validated on the board. The template advisory is explicitly not challenge-compliant Llama inference.

## Proposed RK3588J core assignment

- Network 1D-CNN → `RKNNLite.NPU_CORE_0`
- URL character CNN → `RKNNLite.NPU_CORE_1`
- SMS character CNN → `RKNNLite.NPU_CORE_2`
- Llama via ExecuTorch → Arm CPU

The adapter creates an independent RKNNLite runtime context for each model and requests the matching core mask. This is a target allocation, not proof of simultaneous physical execution. Driver scheduling, memory bandwidth, context contention and thermal effects must be measured. The concurrency benchmark harness measures concurrent host requests; use available driver counters or vendor profiling to establish hardware overlap.

## Model lifecycle

1. Preserve v1.3.0 Item 2 `.joblib` reference models and their reports.
2. Prepare labelled CSVs for each compact model. Network CSV must use exactly `app.ml.flow_reducer.MODEL_FEATURES` plus `label`; URL CSV uses `url,label`; SMS CSV uses `message,label`.
3. Train candidate PyTorch models with `scripts/edge_models/train_compact.py`.
4. Use group-aware splits for final evaluation: URL by registrable domain and network by capture day/campaign. The script's random stratified split is only a bring-up baseline and must not be cited as final research evaluation.
5. Calibrate thresholds on a separate validation set, not the test set. Review precision, recall, F1, FPR, PR-AUC, confusion matrix and confidence intervals.
6. Export ONNX and convert to RKNN INT8 with a compatible RKNN Toolkit2 installation and representative calibration data.
7. Assemble `models/rknn/manifest.json` with `scripts/edge_models/build_manifest.py`.
8. Run board inference, CPU-vs-NPU parity, per-model latency and three-way concurrent tests.
9. Only promote a candidate after independent held-out validation and physical board verification.

### Network feature contract

The 26 input columns are imported directly from `MODEL_FEATURES` in `app/ml/flow_reducer.py`. Do not recreate or reorder the list manually. Training records mean/std in the training manifest; the deployment manifest must carry those same values so runtime normalisation matches training.

### URL/SMS character encoding

The training and inference code share `app/edge_models/preprocess.py`. Unknown characters map to UNK; fixed length is 512 by default; inputs are one-hot float32 tensors. Changing alphabet, casing, truncation or max length requires retraining and a new model manifest version.

## Runtime API

- `GET /api/hardware/rknn/status` — per-model readiness, requested core, artifact and initialisation error.
- `POST /api/hardware/rknn/network` — `{"features": { ...26 feature keys... }}`.
- `POST /api/hardware/rknn/url` — `{"url":"https://..."}`.
- `POST /api/hardware/rknn/sms` — `{"message":"..."}`.

These endpoints are explicit hardware inference probes. The existing public analysis endpoints continue to use the v1.3.0 detector paths unless a future release promotes and validates the NPU models into the primary path. An unavailable or missing RKNN artifact returns an explicit 503; it does not pretend to have run on the NPU.

## Install and run

Base application (existing project venv):

```bash
python -m pip install -e .
python -m pip install -e '.[network,dev]'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Training workstation (separate environment recommended):

```bash
python -m pip install -e '.[training]'
python scripts/edge_models/train_compact.py network data/training/network_features.csv
python scripts/edge_models/train_compact.py url data/training/urls.csv
python scripts/edge_models/train_compact.py sms data/training/sms.csv
```

The supplied training script writes ONNX and PyTorch checkpoints under `models/torch`. The current SMS training baseline is the character CNN. The BiGRU class is available for CPU experiments; it is not yet wired into the training/export command.

RKNN Toolkit2 conversion should be run in a vendor-supported conversion environment. Example:

```bash
python scripts/edge_models/convert_to_rknn.py models/torch/network_compact_fp32.onnx \
  --dataset calibration/network_calibration.txt --output models/rknn/network_1dcnn_int8.rknn
# Repeat for url and sms with representative calibration tensors/datasets.
python scripts/edge_models/build_manifest.py
```

Check that the installed Toolkit2 supports calibration inputs for these non-image tensor shapes. The conversion helper is a starting adapter; exact dataset formatting and quantisation configuration are SDK-version-specific.

On EdgeBLE board, use the already working Debian system Python/RKNNLite environment if that is where `rknnlite` is installed. Do not force RKNNLite into the project venv or Miniforge Python if the vendor wheel is incompatible. A subprocess bridge can be added if the API environment and RKNNLite environment must remain separate.

## Required validation gates

- Confirm model artifact SHA-256, input shape, feature order, alphabet and threshold in manifest.
- Verify each model initialises on its assigned core.
- Verify known test vectors against PyTorch/ONNX outputs within a documented tolerance.
- Run each detector independently, then all three concurrently.
- Run Llama ExecuTorch concurrently with the three detector workloads.
- Record median/p95 latency, throughput, CPU/RAM, temperature, NPU utilisation if the driver exposes a real counter, and power if measured.
- Mark unavailable metrics as unavailable, never as 0.
- Run offline test with network disconnected.

## Current status and honest claims

Implemented in this source package: compact PyTorch model definitions, deterministic character preprocessing, network feature contract, RKNNLite per-core adapter, explicit inference/status API probes, training/export baseline scripts, manifest builder and concurrency harness.

Not included / not yet proven: trained compact model checkpoints, ONNX exports, RKNN `.rknn` artifacts, quantisation calibration corpus, model threshold calibration, physical inference on EdgeBLE, simultaneous three-core scheduling, Llama `.pte`/tokenizer/generation loop, ExecuTorch board runtime and end-to-end offline jury demo. The Item 2 `.joblib` artefacts remain CPU/scikit-learn models and are not automatically converted by this release.
