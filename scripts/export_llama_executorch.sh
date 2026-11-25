#!/usr/bin/env bash
# ==============================================================================
# EdgeShield AI — Reproducible Llama-3.2-1B-Instruct -> ExecuTorch (.pte) Export
# Target: Arm64 Cortex-A CPU with XNNPACK + Arm KleidiAI
# Quantization: 8da4w (linear group_size=128) + 4-bit embeddings (group_size=32)
# Verified RK3588J output: 766 MB (.pte), ~1.01 GB RSS, max_context_len=512
# ==============================================================================
set -euo pipefail

MODEL_DIR="${1:-./models/Llama-3.2-1B-Instruct}"
OUTPUT_DIR="${2:-./models/exported}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
EMB_QUANT="${EMB_QUANT:-4,32}"
GROUP_SIZE="${GROUP_SIZE:-128}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Error: Model directory '$MODEL_DIR' not found." >&2
  echo "Download Meta Llama-3.2-1B-Instruct weights first." >&2
  exit 1
fi

CHECKPOINT=""
PARAMS=""
TOKENIZER=""
if [[ -f "${MODEL_DIR}/original/consolidated.00.pth" ]]; then
  CHECKPOINT="${MODEL_DIR}/original/consolidated.00.pth"
  PARAMS="${MODEL_DIR}/original/params.json"
  TOKENIZER="${MODEL_DIR}/tokenizer.json"
  [[ -f "$TOKENIZER" ]] || TOKENIZER="${MODEL_DIR}/original/tokenizer.model"
elif [[ -f "${MODEL_DIR}/consolidated.00.pth" ]]; then
  CHECKPOINT="${MODEL_DIR}/consolidated.00.pth"
  PARAMS="${MODEL_DIR}/params.json"
  TOKENIZER="${MODEL_DIR}/tokenizer.json"
  [[ -f "$TOKENIZER" ]] || TOKENIZER="${MODEL_DIR}/tokenizer.model"
else
  echo "Error: No consolidated.00.pth found in '$MODEL_DIR'." >&2
  exit 1
fi

OUT_NAME="llama_3_2_1b_xnnpack_ctx${MAX_SEQ_LEN}.pte"
(
  cd "$OUTPUT_DIR"
  python3 -m executorch.examples.models.llama.export_llama     --model llama3_2     --checkpoint "${CHECKPOINT}"     --params "${PARAMS}"     -kv     --use_sdpa_with_kv_cache     -X     -qmode 8da4w     -G "${GROUP_SIZE}"     -E "${EMB_QUANT}"     --max_seq_length "${MAX_SEQ_LEN}"     --max_context_length "${MAX_SEQ_LEN}"     --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'     --output_name "${OUT_NAME}"
)

if [[ -f "${TOKENIZER}" ]]; then
  cp "${TOKENIZER}" "${OUTPUT_DIR}/tokenizer.json"
fi
echo "Exported -> ${OUTPUT_DIR}/${OUT_NAME} (max_context_len=${MAX_SEQ_LEN}, qmode=8da4w G=${GROUP_SIZE}, emb=${EMB_QUANT})"
