# scripts/run_S4_augmentation.sh   – basic vs strong vs extra‑strong
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# assumes ct_transform.py reads $AUG_STRENGTH={basic|strong|xstrong}
for AUG in basic strong xstrong; do
  AUG_STRENGTH=$AUG torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
    --csv $DATA_CSV --pretrained $PRETRAIN_CKPT \
    --batch-size 8 --epochs 60 \
    --lr 2e-4 --lr-backbone-mult 0.1 \
    --freeze-epochs 3 --balance-sampler --amp \
    --use-ema --use-swa \
    --output $RUN_ROOT/S4_aug_${AUG} \
    --wandb-project $WANDB_PROJECT \
    --wandb-name   "${WANDB_NAME_PREFIX}_S4_aug_${AUG}"
done
