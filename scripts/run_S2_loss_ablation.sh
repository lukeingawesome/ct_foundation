# scripts/run_S2_loss_ablation.sh   – focal‑γ grid (2 runs)
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
for GAMMA in 1.5 2.5; do
  torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
    --csv $DATA_CSV --pretrained $PRETRAIN_CKPT \
    --batch-size 8 --epochs 60 \
    --lr 2e-4 --lr-backbone-mult 0.1 \
    --focal-gamma $GAMMA \
    --balance-sampler --amp \
    --use-ema --use-swa \
    --output $RUN_ROOT/S2_loss_gamma${GAMMA//./} \
    --wandb-project $WANDB_PROJECT \
    --wandb-name   "${WANDB_NAME_PREFIX}_S2_loss_gamma${GAMMA//./}"
done
