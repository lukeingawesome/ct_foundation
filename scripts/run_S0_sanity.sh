# scripts/run_S0_sanity.sh   – 3‑epoch smoke test (~40 min)
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
torchrun --nproc_per_node=4 $PYTHON_SCRIPT \
  --csv $DATA_CSV \
  --pretrained $PRETRAIN_CKPT \
  --batch-size 8 \
  --epochs 3 \
  --lr 2e-4 \
  --lr-backbone-mult 0.1 \
  --freeze-epochs 0 \
  --balance-sampler \
  --amp \
  --use-ema \
  --output $RUN_ROOT/S0_sanity
  
