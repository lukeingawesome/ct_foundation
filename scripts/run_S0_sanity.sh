# scripts/run_S0_sanity.sh   – 3‑epoch smoke test (~2 min with 1% data)
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
  --csv $DATA_CSV \
  --pretrained $PRETRAIN_CKPT \
  --batch-size 8 \
  --epochs 3 \
  --lr 2e-4 \
  --lr-backbone-mult 0.1 \
  
  --balance-sampler \
  --amp \
  --use-ema \
  --train-data-fraction 0.01 \
  --output $RUN_ROOT/S0_sanity
  
