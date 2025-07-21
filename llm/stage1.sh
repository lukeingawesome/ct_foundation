export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=/model/huggingface
export HUGGINGFACE_HUB_TOKEN=
export MASTER_PORT=29555                 # or any other free port

torchrun --standalone \
         --nproc_per_node=2 \
         --rdzv_backend=c10d \
         train_medgemma_stage1.py \
         --csv /data/all_ct_with_labels.csv \
         --ct_ckpt /model/1c_siglip2/pytorch_model.bin \
         --bs 16 \
         --epochs 5