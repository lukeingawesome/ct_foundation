# choose GPUs (example: cards 4‑5)
export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=/model/huggingface
export HUGGINGFACE_HUB_TOKEN=
export MASTER_PORT=29556     # cache

torchrun --standalone --nproc_per_node=2 \
         train_medgemma_stage2.py \
         --csv  llm/all_ct_with_labels_with_answer_cleaned.csv \
         --ct_ckpt  /model/1c_siglip2/pytorch_model.bin \
         --stage1_ckpt  projector_ckpt/best_projector.pt \
         --bs 2  --epochs 2
