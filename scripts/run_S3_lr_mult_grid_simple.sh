# scripts/run_S3_lr_mult_grid_simple.sh   – 3 × 2 grid (6 runs) - SIMPLE VERSION
#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

# Set explicit timeout and retry settings
export NCCL_TIMEOUT=1800                  # 30 minutes timeout
export NCCL_BLOCKING_WAIT=1               # Use blocking wait
export NCCL_DEBUG=INFO                    # Enable debug output

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
    if [ -d "$output_dir" ]; then
        echo "Cleaning up failed run directory: $output_dir"
        rm -rf "$output_dir"
    fi
    
    # Wait a moment for system to settle
    sleep 5
    
    # Run with timeout and error handling
    if timeout 7200 torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
      --csv $DATA_CSV --pretrained $PRETRAIN_CKPT \
      --batch-size 8 --epochs 60 \
      --lr $LR --lr-backbone-mult $M \
      --balance-sampler --amp \
      --use-ema --use-swa \
      --output "$output_dir" \
      --wandb-project $WANDB_PROJECT \
      --wandb-name "${WANDB_NAME_PREFIX}_S3_lr${LR}_mult${M}" \
      --patience 8 \
      --log-level INFO; then
        echo "✅ Run completed successfully: LR=$LR, MULT=$M"
    else
        echo "❌ Run failed: LR=$LR, MULT=$M"
        echo "   This might be due to NCCL timeout or other distributed training issues."
        echo "   Consider running the diagnostic script: ./scripts/diagnose_distributed.sh"
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