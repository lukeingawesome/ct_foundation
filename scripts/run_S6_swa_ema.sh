# scripts/run_S6_swa_ema.sh   – {EMA|SWA|both|none}
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
declare -a COMBOS=("ema" "swa" "both" "none")
for C in "${COMBOS[@]}"; do
  USE_EMA=false; USE_SWA=false
  [[ $C == "ema"  || $C == "both" ]] && USE_EMA=true
  [[ $C == "swa"  || $C == "both" ]] && USE_SWA=true
  torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
    --csv $DATA_CSV --pretrained $PRETRAIN_CKPT \
    --batch-size 8 --epochs 60 \
    --lr 2e-4 --lr-backbone-mult 0.1 \
    --balance-sampler --amp \
    $( $USE_EMA && echo "--use-ema" ) \
    $( $USE_SWA && echo "--use-swa" ) \
    --output $RUN_ROOT/S6_${C} \
    --wandb-project $WANDB_PROJECT \
    --wandb-name   "${WANDB_NAME_PREFIX}_S6_${C}"
done
