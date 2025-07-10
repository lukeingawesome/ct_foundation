# scripts/run_S5_freeze_schedule.sh   – NO LONGER FUNCTIONAL
# NOTE: Gradual unfreezing has been removed from the training script.
# All parameters are now trainable from the start with discriminative LR.
# This script is kept for reference but will not produce the intended behavior.
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"

echo "WARNING: This script is no longer functional since gradual unfreezing was removed."
echo "All parameters are now trainable from the start with discriminative learning rates."
exit 1

# ORIGINAL (NOW DISABLED):
# for FZ in 0 3 6; do
#   torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
#     --csv $DATA_CSV --pretrained $PRETRAIN_CKPT \
#     --batch-size 8 --epochs 60 \
#     --lr 2e-4 --lr-backbone-mult 0.1 \
#     --freeze-epochs $FZ --balance-sampler --amp \
#     --use-ema --use-swa \
#     --output $RUN_ROOT/S5_freeze${FZ} \
#     --wandb-project $WANDB_PROJECT \
#     --wandb-name   "${WANDB_NAME_PREFIX}_S5_freeze${FZ}"
# done
