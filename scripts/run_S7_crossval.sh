# scripts/run_S7_crossval.sh   – 5‑fold CV (needs fold column in CSV)
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
for FOLD in 0 1 2 3 4; do
  torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
    --csv $DATA_CSV \
    --pretrained $PRETRAIN_CKPT \
    --batch-size 8 --epochs 60 \
    --lr 2e-4 --lr-backbone-mult 0.1 \
    --balance-sampler --amp \
    --use-ema --use-swa \
    --fold $FOLD \
    --output $RUN_ROOT/S7_cv_f$FOLD \
    --wandb-project $WANDB_PROJECT \
    --wandb-name   "${WANDB_NAME_PREFIX}_S7_cv_f$FOLD"
done
