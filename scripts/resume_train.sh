#!/usr/bin/env bash
# --------------------------- configurable section ----------------------------
MEM_THRESHOLD=3000         # MiB – ignore GPUs using more than this much memory
SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/env.sh"

# The first (optional) CLI argument = requested GPU count
REQ_GPUS="${1:-0}"         # 0 ⇒ use *all* GPUs that pass the threshold check
# --------------------------------------------------------------------------- #

# Function: discover all GPUs whose memory.used is below $MEM_THRESHOLD
discover_free_gpus () {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
  awk -F',' -v thr="$MEM_THRESHOLD" '
       {gsub(/ /, "", $1); gsub(/ /, "", $2)}
       ($2 < thr) {printf "%s ", $1}'              # space‑separated list
}

read -ra FREE_GPUS <<< "$(discover_free_gpus)"
FREE_COUNT="${#FREE_GPUS[@]}"

# DEBUG: Show what GPUs were discovered
echo "[DEBUG] Memory threshold: ${MEM_THRESHOLD} MiB"
echo "[DEBUG] Free GPUs discovered: ${FREE_GPUS[*]}"
echo "[DEBUG] Free GPU count: ${FREE_COUNT}"

# If user didn't specify a number, take everything we found
if [[ "$REQ_GPUS" -eq 0 ]]; then
    REQ_GPUS="$FREE_COUNT"
fi

echo "[DEBUG] Requested GPUs: ${REQ_GPUS}"

# Sanity checks ----------------------------------------------------------------
if [[ "$FREE_COUNT" -eq 0 ]]; then
  echo "[ERROR] No GPUs have < ${MEM_THRESHOLD} MiB in‑use. Abort." >&2
  exit 1
fi

if [[ "$REQ_GPUS" -gt "$FREE_COUNT" ]]; then
  echo "[ERROR] Requested $REQ_GPUS GPU(s), but only $FREE_COUNT remain below "\
       "${MEM_THRESHOLD} MiB. Reduce the request or free GPUs." >&2
  exit 1
fi
# ------------------------------------------------------------------------------

# Pick the first $REQ_GPUS from FREE_GPUS and export CUDA_VISIBLE_DEVICES
SELECTED_GPU_LIST="$(printf "%s," "${FREE_GPUS[@]:0:REQ_GPUS}")"
CUDA_VISIBLE_DEVICES="${SELECTED_GPU_LIST%,}"   # strip trailing comma

# DEBUG: Show final selection
echo "[DEBUG] Selected GPU list: ${SELECTED_GPU_LIST}"
echo "[DEBUG] Final CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

export CUDA_VISIBLE_DEVICES

echo "[INFO] Using GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] nproc_per_node set to ${REQ_GPUS}"

# ------------------------------ launch ----------------------------------------
# Check for existing checkpoint to resume from
LAST_CKPT="$RUN_ROOT/S1_baseline_3ch/last_epoch.pth"
[[ -f "$LAST_CKPT" ]] && RESUME="--resume $LAST_CKPT" || RESUME=""

echo "[INFO] Checkpoint path: ${LAST_CKPT}"
if [[ -n "$RESUME" ]]; then
    echo "[INFO] Found existing checkpoint, will resume training"
else
    echo "[INFO] No existing checkpoint found, starting fresh training"
fi

torchrun --nproc_per_node="${REQ_GPUS}" $PYTHON_SCRIPT \
  --csv "$DATA_CSV" \
  --pretrained "$PRETRAIN_CKPT" \
  $RESUME \
  --batch-size 8 \
  --epochs 60 \
  --lr 2e-4 \
  --lr-backbone-mult 0.1 \
  --balance-sampler \
  --amp \
  --use-ema \
  --use-swa \
  --focal-gamma 0.0 \
  --output "$RUN_ROOT/S1_baseline_3ch" \
  --three-channel \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name   "${WANDB_NAME_PREFIX}_S1_baseline_learning_all_3ch"
