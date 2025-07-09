#!/bin/bash

# Fine-tune SigLIP on chest CT multilabel classification
# Usage: ./run_finetune.sh

python finetune_siglip.py \
  --csv /data/all_ct_with_labels.csv \
  --pretrained /model/1c_siglip/pytorch_model.bin \
  --batch-size 16 \
  --epochs 60 \
  --lr 1e-4 \
  --lr-backbone-mult 0.05 \
  --balance-sampler \
  --use-ema \
  --use-swa \
  --amp 