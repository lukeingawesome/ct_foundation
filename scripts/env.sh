# scripts/env.sh
#!/usr/bin/env bash
set -euo pipefail

# ————————————————————————————————————————————————————
# EDIT THESE PATHS ONCE
export DATA_CSV="/data/all_ct_with_labels.csv"
export PRETRAIN_CKPT="/model/1c_siglip2/pytorch_model.bin"
export PYTHON_SCRIPT="finetune_siglip_ct.py"
export RUN_ROOT="runs"                # all outputs will live here
export WANDB_PROJECT="siglip-ct"
export WANDB_NAME_PREFIX="$(hostname -s)_$(date +%y%m%d%H%M)"
# Optionally set your WANDB key:
# export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxx"
# ————————————————————————————————————————————————————

# Harden CUDA launch
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=8      # keeps dataloader snappy
