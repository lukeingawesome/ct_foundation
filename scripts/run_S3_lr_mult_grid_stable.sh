# scripts/run_S3_lr_mult_grid_stable.sh   – 3 × 2 grid (6 runs) - STABLE VERSION
#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

# Additional safety measures for distributed training
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Function to check if previous run failed and clean up
cleanup_failed_run() {
    local output_dir="$1"
    if [ -d "$output_dir" ]; then
        echo "Cleaning up failed run directory: $output_dir"
        rm -rf "$output_dir"
    fi
}

# Function to wait for GPUs to be available
wait_for_gpus() {
    echo "Waiting for GPUs to be available..."
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r gpu_id mem_used mem_total; do
        echo "GPU $gpu_id: ${mem_used}MB / ${mem_total}MB used"
    done
    sleep 5
}

declare -a LRS=(1e-4 2e-4 3e-4)
declare -a MULTS=(0.05 0.1)

for LR in "${LRS[@]}"; do
  for M in "${MULTS[@]}"; do
    output_dir="$RUN_ROOT/S3_lr${LR}_mult${M}"
    
    echo "=========================================="
    echo "Starting run: LR=$LR, MULT=$M"
    echo "Output directory: $output_dir"
    echo "=========================================="
    
    # Clean up any failed previous run
    cleanup_failed_run "$output_dir"
    
    # Wait for GPUs to be available
    wait_for_gpus
    
    # Run with additional safety measures
    torchrun \
      --nproc_per_node=4 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=127.0.0.1:29500 \
      --rdzv_id=grid_search \
      $PYTHON_SCRIPT \
        --csv $DATA_CSV --pretrained $PRETRAIN_CKPT \
        --batch-size 8 --epochs 30 \
        --lr $LR --lr-backbone-mult $M \
        --balance-sampler --amp \
        --use-ema --use-swa \
        --output "$output_dir" \
        --wandb-project $WANDB_PROJECT \
        --wandb-name "${WANDB_NAME_PREFIX}_S3_lr${LR}_mult${M}" \
        --patience 8 \
        --log-level INFO
    
    # Check if run completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Run completed successfully: LR=$LR, MULT=$M"
    else
        echo "❌ Run failed: LR=$LR, MULT=$M"
        # Don't exit, continue with next run
    fi
    
    # Wait between runs to ensure clean state
    echo "Waiting 30 seconds before next run..."
    sleep 30
    
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
  done
done

echo "=========================================="
echo "Grid search completed!"
echo "==========================================" 