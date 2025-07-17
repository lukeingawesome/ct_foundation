# scripts/run_S3_lr_mult_grid.sh   – 3 × 2 grid (6 runs)
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
declare -a LRS=(1e-4 2e-4 3e-4)
declare -a MULTS=(0.05 0.1)
for LR in "${LRS[@]}"; do
  for M in "${MULTS[@]}"; do
    torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
      --csv $DATA_CSV --pretrained $PRETRAIN_CKPT \
      --batch-size 8 --epochs 60 \
      --lr $LR --lr-backbone-mult $M \
      --balance-sampler --amp \
      --use-ema --use-swa \
      --output $RUN_ROOT/S3_lr${LR}_mult${M} \
      --wandb-project $WANDB_PROJECT \
      --wandb-name   "${WANDB_NAME_PREFIX}_S3_lr${LR}_mult${M}"
  done
done
