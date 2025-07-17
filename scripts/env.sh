# scripts/env.sh
#!/usr/bin/env bash
set -euo pipefail

# ————————————————————————————————————————————————————
# EDIT THESE PATHS ONCE
export DATA_CSV="/data/all_ct_with_labels.csv"
export PRETRAIN_CKPT="/model/3c_siglip/pytorch_model.bin"
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

# Enhanced NCCL configuration for stability
export NCCL_DEBUG=INFO                    # Enable NCCL debug output
export NCCL_TIMEOUT=1800                  # Increase timeout to 30 minutes
export NCCL_IB_TIMEOUT=1800               # InfiniBand timeout
export NCCL_SOCKET_TIMEOUT=1800           # Socket timeout
export NCCL_BLOCKING_WAIT=1               # Use blocking wait
export NCCL_ASYNC_ERROR_HANDLING=1        # Enable async error handling
export NCCL_P2P_DISABLE=0                 # Enable P2P communication
export NCCL_IB_DISABLE=0                  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=lo              # Use loopback interface for local training
export NCCL_IB_HCA=mlx5_0                 # Specify IB HCA if available

# Additional stability settings
export CUDA_LAUNCH_BLOCKING=0             # Non-blocking CUDA operations
export TORCH_DISTRIBUTED_DEBUG=DETAIL     # Enable detailed distributed debugging
export TORCH_SHOW_CPP_STACKTRACES=1       # Show C++ stack traces on errors

# Memory and performance optimizations
export NCCL_BUFFSIZE=8388608              # 8MB buffer size
export NCCL_NTHREADS=8                    # Number of NCCL threads
export NCCL_RINGS=4                       # Number of NCCL rings

# Fallback settings for problematic environments
export NCCL_P2P_NET_CHUNKSIZE=2097152    # 2MB chunk size for P2P
export NCCL_SHM_DISABLE=0                 # Enable shared memory
export NCCL_NET_GDR_LEVEL=0               # Disable GPU Direct RDMA if problematic
