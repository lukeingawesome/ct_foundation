# scripts/infer_TTA.sh   – inference with 6‑view TTA using best.ckpt
#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
CKPT="$RUN_ROOT/S1_baseline/best.pth"   # ← path of model to test
python infer_with_tta.py \
  --csv $DATA_CSV \
  --checkpoint $CKPT \
  --tta-views 6 \
  --batch-size 16 \
  --output $RUN_ROOT/TTA_preds.npy
