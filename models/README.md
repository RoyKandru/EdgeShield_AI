# Model Artifacts

## Included in `EdgeShield_latest.zip`
- `models/rknn/` — All 3 validated RK3588 FP16 NPU models (`network_ids_fp16.rknn`, `url_hybrid_fp16.rknn`, `sms_bigru_fp16.rknn` + `sms_bigru_emb.npz` + `*_meta.json` + `manifest.json`, ~5.2 MB total).
- `models/*_v2*.joblib` — Classical ML fallback / hybrid models for URL, SMS, and Network detection.

## Excluded from `EdgeShield_latest.zip` (Add Separately for Pipeline B Llama Advisory)
Because the ExecuTorch Llama-3.2-1B model is ~1.69 GB, it is intentionally omitted from the shareable zip. To enable on-device Llama advisory:

1. Place your ExecuTorch-exported `.pte` file and matching tokenizer here:
   - `models/llama.pte` (or `models/llama_3_2_1b_xnnpack.pte`)
   - `models/tokenizer.model` (or `models/tokenizer.json`)
2. Restart EdgeShield (`./scripts/run_board.sh`). It will automatically detect the weights and start `bin/executorch_runner --daemon`.

Without `llama.pte`, EdgeShield automatically runs with its built-in deterministic `template` advisory fallback while all 3 RK3588 NPU detectors and SCI run normally.
