export CUDA_VISIBLE_DEVICES=5
python3 finetune_siglip_ct2.py \
  --ckpt /model/ct_siglip/final3.pth \
  --csv  /data/all_ct_with_labels.csv \
  --epochs 0 --lr-head 1e-3 \
  --wandb-project ct-siglip \
  --wandb-name   stage2_try2