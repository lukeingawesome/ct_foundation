#!/usr/bin/env python3
# fast_eval_stage2.py
# ------------------------------------------------------------
# Usage (single GPU):
#   CUDA_VISIBLE_DEVICES=0 python fast_eval_stage2.py \
#        --csv data/all_ct_with_labels.csv \
#        --ct_ckpt ckpts/merlin.pth \
#        --stage1_ckpt projector_ckpt/best_projector.pt \
#        --stage2_ckpt stage2_lora_ckpt/stage2_ep1.pt \
#        --out_csv stage2_val_predictions.csv \
#        --bs 4
# ------------------------------------------------------------
import os, gc, csv, re, math, argparse, logging
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data     import Dataset, DataLoader
from torchvision.transforms._presets import VideoClassification  # tiny normalise
from tqdm import tqdm

# -------- 3rd‑party metrics (vectorised) --------------------
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer, scoring
from nltk.translate.meteor_score import meteor_score
# ------------------------------------------------------------

from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          set_seed, logging as hf_logging,
                          GenerationConfig, AutoTokenizer,
                          )
from transformers.modeling_outputs import BaseModelOutput

# silence HF info spam
hf_logging.set_verbosity_error()

# -------- PEFT for LoRA loading ------------------------------
from peft import LoraConfig, get_peft_model
# ------------------------------------------------------------

# -------- local modules ------------------------------------
from merlin                import Merlin
from train_medgemma_stage1 import CTProjector
from training.ct_transform import get_val_transform
from data_chat             import _hu_window_to_unit
# ------------------------------------------------------------

# ---------- helper --------------------------------------------------------
TAG_RE = re.compile(r"<.*?>")

def _clean(s: str) -> str:
    "strip tags + redundant spaces"
    return TAG_RE.sub("", s).strip()

def _tokenise(s: str):
    return _clean(s).lower().split()

# ---------- dataset -------------------------------------------------------
import pandas as pd, numpy as np
class CTEvalDS(Dataset):
    def __init__(self, csv_path: str, transform, three_ch=False):
        self.df = pd.read_csv(csv_path).query("split=='val'").reset_index(drop=True)
        self.tr = transform; self.three = three_ch

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        vol = np.load(r.img_path)["image"]          # (C,D,H,W)  or (1,D,H,W)
        if self.three:
            vol = vol*2500. - 1000.
            lung = _hu_window_to_unit(vol[0], -600,1000)
            medi = _hu_window_to_unit(vol[0],   40, 400)
            bone = _hu_window_to_unit(vol[0],  700,1500)
            vol  = np.stack([lung,medi,bone],0)
        img = torch.from_numpy(vol).float()
        if self.tr: img = self.tr(img)

        return dict(img=img,
                    instr=str(r.instruction),
                    ref=str(r.answer),
                    path=r.img_path)

# ---------- collate -------------------------------------------------------
from PIL import Image
DUMMY_RGB = Image.new("RGB",(8,8))
def collate(batch, processor):
    imgs   = [b["img"]   for b in batch]
    instrs = [b["instr"] for b in batch]

    # user message  ➜  <task=report>  <image>  instruction
    msgs = [[{"role":"user",
              "content":[{"type":"text","text":"<task=report>"},
                         {"type":"image"},
                         {"type":"text","text":instrs[i]}]}]
            for i in range(len(batch))]

    prompts = [processor.apply_chat_template(m, add_generation_prompt=True,
                                             tokenize=False).strip()
               for m in msgs]
    enc = processor(text=prompts,
                    images=[[DUMMY_RGB]]*len(prompts),
                    padding=True, truncation=True, max_length=768,
                    return_tensors="pt")
    enc["pixel_values"] = torch.stack(imgs)
    # metadata to return unchanged
    meta = [{k:v for k,v in b.items() if k not in ("img",)} for b in batch]
    return enc, meta

# ---------- CT tower  -----------------------------------------------------
class CTTower(nn.Module):
    def __init__(self, enc, proj, hidden):
        super().__init__(); self.enc, self.proj = enc, proj
        self.config = type("cfg", (), {"hidden_size": hidden})

    @torch.no_grad()
    def forward(self, pixel_values, **_):
        _, tok = self.proj(self.enc(pixel_values))   # (B,256,H)
        return BaseModelOutput(last_hidden_state=tok)

# ---------- passthrough projector (same as training) ----------------------
class PassthroughProjector(nn.Module):
    """
    Gemma3Model expects `forward(vision_outputs)` and wants a *tensor*
    of shape (B, 256, hidden).  We simply return the tokens produced
    by our custom tower.
    """
    def forward(self, vision_outputs, *args, **kwargs):
        if isinstance(vision_outputs, torch.Tensor):
            return vision_outputs                       # safety path
        return vision_outputs.last_hidden_state         # usual path

# ---------- load everything ----------------------------------------------
def build_model(args, device):
    # base LLM
    model = AutoModelForImageTextToText.from_pretrained(
                args.model_id, torch_dtype=torch.bfloat16,
                device_map=device, low_cpu_mem_usage=True)

    # processor / tokenizer
    proc  = AutoProcessor.from_pretrained(args.model_id, local_files_only=True)
    proc.tokenizer.padding_side = "right"

    # CT encoder + projector
    enc = Merlin(ImageEmbedding=True)
    ct_state = torch.load(args.ct_ckpt, map_location="cpu")
    
    # Filter out text encoder keys since we only need image encoder
    filtered_state = {k: v for k, v in ct_state.items() 
                     if not k.startswith('encode_text.')}
    
    # Print checkpoint info for debugging
    print(f"CT checkpoint keys: {len(ct_state)}")
    print(f"Filtered keys: {len(filtered_state)}")
    print(f"Text encoder keys removed: {len(ct_state) - len(filtered_state)}")
    
    enc.load_state_dict(filtered_state, strict=False)
    enc.eval(); enc.to(device[""])

    pj_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    proj = CTProjector(pj_ckpt["feat_dim"], pj_ckpt["hidden_dim"])
    proj.load_state_dict(pj_ckpt["projector"])
    proj.eval(); proj.to(device[""])

    tower = CTTower(enc, proj, pj_ckpt["hidden_dim"])
    model.model.vision_tower        = tower
    model.model.multi_modal_projector = PassthroughProjector()

    # --------------- add LoRA modules (exactly as in training) ------------
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)

    # handle DDP module prefixes and load the LoRA weights + updated projector
    state = torch.load(args.stage2_ckpt, map_location="cpu")
    if next(iter(state)).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("⋯ missing", len(missing), "unexpected", len(unexpected))
    
    # Print the actual missing keys
    if missing:
        print("\nMissing keys:")
        for key in missing:
            print(f"  - {key}")
    
    # Print the actual unexpected keys
    if unexpected:
        print("\nUnexpected keys:")
        for key in unexpected:
            print(f"  - {key}")

    model.eval(); return model, proc

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ct_ckpt", "--ct", required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--model_id", default="google/medgemma-4b-it")
    ap.add_argument("--out_csv",  default="stage2_val_predictions.csv")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=800)
    args = ap.parse_args()

    set_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = {"" : "cuda" if torch.cuda.is_available() else "cpu"}
    model, proc = build_model(args, device)

    ds = CTEvalDS(args.csv, get_val_transform(), three_ch=False)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4,
                        collate_fn=lambda b: collate(b, proc))

    END_ID  = proc.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    gen_cfg = GenerationConfig(
                 max_new_tokens=args.max_new,
                 do_sample=False, num_beams=1,
                 pad_token_id=proc.tokenizer.pad_token_id,
                 eos_token_id=END_ID,
                 no_repeat_ngram_size=4,
                 repetition_penalty=1.1,
             )

    refs, hyps, meta_rows = [], [], []

    scorch = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_agg = scoring.BootstrapAggregator()

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch_idx, (enc, meta) in enumerate(tqdm(loader, desc="inference")):
            enc = {k:v.to(device[""]) if torch.is_tensor(v) else v for k,v in enc.items()}
            gen_ids = model.generate(**enc, **gen_cfg.to_dict())

            # remove the user‑prompt portion so we don't decode it
            outs = []
            for ids, inp in zip(gen_ids, enc["input_ids"]):
                prompt_len = (inp != proc.tokenizer.pad_token_id).sum().item()
                outs.append(proc.tokenizer.decode(
                    ids[prompt_len:],
                    skip_special_tokens=True).strip())

            # Log one sample from each batch
            if batch_idx % 10 == 0:  # Log every 10th batch to avoid spam
                sample_idx = 0  # Log first sample from batch
                sample_meta = meta[sample_idx]
                sample_hyp = outs[sample_idx]
                sample_instr = sample_meta["instr"]
                image_id = Path(sample_meta["path"]).stem  # Extract filename without extension
                
                print(f"\n--- Batch {batch_idx} Sample ---")
                print(f"Instruction: {sample_instr}")
                print(f"Image: {image_id}")
                print(f"Output: {sample_hyp}")
                print(f"Reference: {sample_meta['ref']}")
                print("-" * 50)

            for m, hyp in zip(meta, outs):
                refs.append(_clean(m["ref"]))
                hyps.append(_clean(hyp))
                meta_rows.append(m)

    # ----- corpus‑level metrics ---------------------------------------
    bleu  = corpus_bleu(hyps, [refs]).score / 100.0           # 0‑1 range
    for r,h in zip(refs, hyps):
        rouge_agg.add_scores(scorch.score(r, h))
    rouge = rouge_agg.aggregate()
    meteor = sum(meteor_score([_tokenise(r)], _tokenise(h)) for r,h in zip(refs,hyps)) / len(refs)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.info(f"BLEU‑4 : {bleu:.4f}   ROUGE‑L : {rouge['rougeL'].mid.fmeasure:.4f}   METEOR : {meteor:.4f}")

    # ----- write CSV ---------------------------------------------------
    with Path(args.out_csv).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_path","instruction","reference","prediction"])
        for m,h in zip(meta_rows, hyps):
            w.writerow([m["path"], m["instr"], m["ref"], h])

    print("✓ saved →", args.out_csv)
    gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
