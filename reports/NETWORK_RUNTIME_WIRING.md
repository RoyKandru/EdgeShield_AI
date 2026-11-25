# Network v2 runtime wiring: what ships, what was measured, what is not claimed

## What runs
`analyse_pcap(path)` (app/detectors/network.py) now does, when `models/network_model_v2r.joblib` exists and
`EDGESHIELD_ML_DETECTORS` is true:

1. `app/ml/flow_reducer.py` reduces the pcap to CICFlowMeter-style bidirectional flows (5-tuple, 120 s
   timeout keeping orientation, FIN closes the flow, flows with <2 packets not emitted) and adds minute-window
   host features per source IP, destination IP and (src,dst) pair.
2. `app/ml/network_runtime.py` scores them (lazy load, offline, refuses a bundle whose feature list differs
   from `flow_reducer.MODEL_FEATURES`).
3. Verdict per flow = **ML score >= threshold OR a production rule fires**; `model_version = network-hybrid-v2`.
   Missing/corrupt/mismatched artefact, or ML disabled: unchanged rules-only path (`network-rules-v1`).
`analyse_flows(list[dict])` is unchanged (rules only).

## Why the shipped model is not `network_model_v2.joblib`
v2 uses 59 CIC CSV columns. Checked against the CSVs, these cannot be reproduced from raw packets with
verified semantics, so they were removed and the cost **measured** (`reports/NETWORK_V2R.md`, identical splits and
identical training rows):

* TCP flag-count columns (6) and syn_count/syn_ratio/sw_rst_frac: CIC flag counts are neither per-packet counts
  nor presence flags (RST is 0 in every flow; SYN is present in only ~2 % of PortScan flows).
* avg_pkt, pl_mean, pl_std, act_data, min_seg, win_f/win_b: CICFlowMeter off-by-one style artefacts / odd values.
* All 17 packet-size features: CIC reply frames show a 6-byte "payload" (the 60-byte Ethernet minimum frame minus
  headers) while forward SYNs show 0, so the payload accounting cannot be reproduced without the original pcaps and
  would make the model sensitive to the capture point (padded vs unpadded frames).

| Protocol B (train Mon-Thu, test Fri; Bot/PortScan/DDoS unseen) | precision | recall | FPR | FP |
|---|---|---|---|---|
| v2, 59 CIC columns | 1.0000 | 0.9838 | 0.0000 | 5 |
| v2r, 43 reproducible features | 1.0000 | 0.9838 | 0.0000 | 4 |
| **v2r, 26 features (shipped, no packet sizes)** | 0.9998 | 0.9838 | 0.0002 | 71 |

Per-family recall (shipped): DDoS 0.989, PortScan 0.991, Bot 0.003. Protocol A: P 0.9992, R 0.9995, FPR 0.0002.
Cost of dropping the unreproducible features: FPR 0.00 % -> 0.02 % on unseen-family test; recall unchanged.

## Parity testing (tests/test_network_ml.py)
All 43 reducer features (not just the 26 the model uses) are compared against the **actual training code**
(`network_improve.load_file`) on constructed flows whose raw CIC-style columns are computed independently with
numpy: equal within 1e-4. Flow-construction rules (FIN, timeout, orientation, <2 packets, Down/Up = floor) and the
hybrid OR logic have unit tests. What parity does NOT cover: the reducer was never compared against real
CICFlowMeter output on the CIC pcaps (not available), so flow-boundary behaviour is a best-effort reproduction.

## Measured on the development PC (not the RK3588J)
120k packets / 20k flows: scapy parse 10.8k packets/s (dominates), reducer 72k packets/s, inference 3.9 us/flow
batched, 0.85 ms median for a one-flow call; `analyse_pcap` end to end 9.2k packets/s; 0 of 20k ordinary
synthetic TLS flows flagged. Bundle 0.24 MB. Board numbers must be measured separately (requirement 3).

## Known limits
* **Scan size boundary:** scans touching >=1,000 distinct ports per minute are flagged ~100 %; a 200-port scan
  only ~1 %. The model learned the CIC line-rate scan (~1,000 ports, ~43k flows per source-minute). Slow/small
  scans are missed (the production rules miss them too).
* **Bot and web attacks** are not detectable from these features (Bot 0.3 %).
* Single lab network (CIC-IDS2017, known label/flow-construction artefacts); minute-resolution windows; the
  runtime hybrid path is batch per pcap, not a sliding window; one-packet flows (e.g. unanswered SYNs) are not
  emitted, exactly as in training, so an unanswered-SYN scan is invisible to the ML path.
* Payload is not used, so the padding/capture-point issue does not affect the shipped model; timing features
  (IAT, duration) still depend on capture clock quality.
* `ML OR rules v1` adds ~27 false positives per 414k benign flows (rules) on top of the model's.
* Not implemented in the C++ native runtime.
