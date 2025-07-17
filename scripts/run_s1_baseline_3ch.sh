# scripts/run_S1_baseline.sh   – main 60‑epoch reference (~14 h)
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
  --csv $DATA_CSV \
  --pretrained $PRETRAIN_CKPT \
  --batch-size 8 \
  --epochs 60 \
  --lr 2e-4 \
  --lr-backbone-mult 0.1 \
  --balance-sampler \
  --amp \
  --use-ema \
  --use-swa \
  --output $RUN_ROOT/S1_baseline_3ch \
  --three-channel \
  --wandb-project $WANDB_PROJECT \
  --wandb-name   "${WANDB_NAME_PREFIX}_S1_baseline_learning_all_3ch"