HF_HOME=/model/huggingface

CUDA_VISIBLE_DEVICES=4 python3 evaluation_medgemma.py \
       --csv          llm/all_ct_with_labels_with_answer_cleaned.csv \
       --ct_ckpt      /model/1c_siglip2/pytorch_model.bin \
       --stage1_ckpt  projector_ckpt/best_projector.pt \
       --stage2_ckpt  stage2_lora_ckpt/stage2_ep2.pt \
       --model_id     google/medgemma-4b-it \
       --out_csv      results/results_stage2_val.csv \
       --bs           64                # adjust to GPU RAM
