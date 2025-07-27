#!/usr/bin/env python3
# stage2_train_medgemma.py
# ------------------------------------------------------------
# Fine‑tune google/medgemma‑4b‑it on CT volumes (findings reports)
# using a CT encoder + projector and LoRA in the Gemma‑3 backbone.
# ------------------------------------------------------------
import os, gc, math, argparse, logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch, torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, set_seed,
    get_cosine_schedule_with_warmup
)
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# -------- local modules -----------------------------------------------
from merlin                import Merlin                 # 3‑D SigLIP encoder
from train_medgemma_stage1 import CTProjector            # projector class
from training.ct_transform import get_train_transform, get_val_transform
# ----------------------------------------------------------------------

# ───────────────────────────────────────────────────────────────────────
#                      Dataset (CSV → CT tensor)
# ───────────────────────────────────────────────────────────────────────
import pandas as pd, numpy as np
@dataclass
class CTRecord:
    vol:  torch.Tensor
    instr:str
    ans:  str

class CTChatDataset(Dataset):
    IMG_COL  = "img_path"
    Q_COL    = "instruction"
    A_COL    = "answer"
    SPLIT_COL= "split"

    def __init__(self, csv_path: str, split: str, transform, three_ch=False):
        self.df = pd.read_csv(csv_path).query(f"{self.SPLIT_COL}=='{split}'").reset_index(drop=True)
        self.tr = transform; self.three = three_ch

    def __len__(self): return len(self.df)

    def __getitem__(self, idx) -> CTRecord:
        r   = self.df.iloc[idx]
        vol = np.load(r[self.IMG_COL])["image"]            # (C,D,H,W)
        if self.three:                                     # optional 3‑chan HU windows
            from data_chat import _hu_window_to_unit
            vol = vol*2500. - 1000.
            lung = _hu_window_to_unit(vol[0], -600,1000)
            medi = _hu_window_to_unit(vol[0],   40, 400)
            bone = _hu_window_to_unit(vol[0],  700,1500)
            vol  = np.stack([lung,medi,bone],0)
        vol = torch.from_numpy(vol).float()
        if self.tr: vol = self.tr(vol)
        return CTRecord(vol, str(r[self.Q_COL]), str(r[self.A_COL]))


# ───────────────────────────────────────────────────────────────────────
#                    Collate  (chat template + masking)
# ───────────────────────────────────────────────────────────────────────
from PIL import Image
_DUMMY_RGB = Image.new("RGB",(8,8))     # placeholder – we feed real CT tensor

class BatchBuilder:
    def __init__(self, processor: AutoProcessor):
        self.pro = processor
        tok = self.pro.tokenizer
        self.pad_id   = tok.pad_token_id
        self.sot_id   = tok.convert_tokens_to_ids("<start_of_turn>")
        self.eot_id   = tok.convert_tokens_to_ids("<end_of_turn>")

    # ---- helper: mask everything up to & incl. the "<start> model" tokens
    def _mask_prompt(self, ids: torch.Tensor) -> torch.Tensor:
        lab = ids.clone()
        starts = (ids == self.sot_id).nonzero(as_tuple=True)[0]
        if len(starts) < 2:
            lab.fill_(-100); return lab
        lab[:starts[1] + 2] = -100               # +2 covers "<sot>,model"
        lab[ids == self.pad_id] = -100
        if hasattr(self.pro, "image_token_id"):
            lab[ids == self.pro.image_token_id] = -100
        return lab

    def __call__(self, batch: List[CTRecord]) -> Dict[str,Any]:
        # chat message per sample
        msgs, cts, imgs = [], [], []
        for b in batch:
            msgs.append([
                {"role":"user",
                 "content":[
                     {"type":"text",  "text":"<task=report>"},
                     {"type":"image"},
                     {"type":"text",  "text":b.instr.strip()}
                 ]},
                {"role":"model",
                 "content":[{"type":"text","text": b.ans.strip()}]},
            ])
            imgs.append(_DUMMY_RGB)        # just to insert <image> token
            cts.append(b.vol)

        # build input_ids / attn etc.
        enc = self.pro(
            text   =[self.pro.apply_chat_template(m, add_generation_prompt=False,
                                                  tokenize=False).strip()
                     for m in msgs],
            images=[[img] for img in imgs],        # list of list
            padding=True, truncation=True, max_length=768,
            return_tensors="pt"
        )
        enc["labels"] = torch.stack([
            self._mask_prompt(ids) for ids in enc["input_ids"]
        ])
        enc["pixel_values"] = torch.stack(cts)     # (B,C,D,H,W)
        return enc


# ───────────────────────────────────────────────────────────────────────
#               Vision‑tower wrapper (CT encoder + projector)
# ───────────────────────────────────────────────────────────────────────
class CTVisionTower(nn.Module):
    def __init__(self, enc: nn.Module, proj: CTProjector, hidden: int):
        super().__init__(); self.enc, self.proj = enc, proj
        self.config = type("cfg", (), {"hidden_size": hidden})
    def forward(self, pixel_values, **_):
        # freeze the CT encoder but keep gradients for the projector
        with torch.no_grad():
            feats = self.enc(pixel_values)          # (B, N, C)
        _, tok = self.proj(feats)                   # projector is trainable
        return BaseModelOutput(last_hidden_state=tok)

class PassthroughProjector(nn.Module):
    def forward(self, vision_outputs, *_, **__):
        if isinstance(vision_outputs, torch.Tensor):
            return vision_outputs
        return vision_outputs.last_hidden_state


# ───────────────────────────────────────────────────────────────────────
#                               main
# ───────────────────────────────────────────────────────────────────────
def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--csv", required=True)
    cli.add_argument("--stage1_ckpt", required=True)  # best_projector.pt
    cli.add_argument("--ct_ckpt",    required=True)   # Merlin weights
    cli.add_argument("--model_id",   default="google/medgemma-4b-it")
    cli.add_argument("--out_dir",    default="stage2_lora_ckpt")
    cli.add_argument("--bs",         type=int, default=4)
    cli.add_argument("--epochs",     type=int, default=2)
    cli.add_argument("--lr",         type=float, default=2e-4)
    cli.add_argument("--workers",    type=int, default=4)
    args = cli.parse_args()

    # ---------- distributed ------------------------------------------------
    distributed = "RANK" in os.environ
    local_rank  = int(os.getenv("LOCAL_RANK",0))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank0  = not distributed or dist.get_rank()==0

    if rank0:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(42 + local_rank)

    # ---------- CT encoder & projector ------------------------------------
    ct_enc = Merlin(ImageEmbedding=True)
    ct_enc.load_state_dict(torch.load(args.ct_ckpt, map_location="cpu"), strict=False)
    ct_enc.eval().requires_grad_(False).to(device)

    pj_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    proj = CTProjector(pj_ckpt["feat_dim"], pj_ckpt["hidden_dim"])
    proj.load_state_dict(pj_ckpt["projector"])
    proj.to(device)

    hidden_dim = pj_ckpt["hidden_dim"]

    # ---------- MedGemma‑IT -----------------------------------------------
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                token=token,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{local_rank}"},
                low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model_id, token=token)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.model_max_length = 1500

    # swap tower + projector
    tower = CTVisionTower(ct_enc, proj, hidden_dim)
    model.vision_tower                  = tower              # convenience
    model.model.vision_tower            = tower              # Gemma3Model uses this
    model.model.multi_modal_projector   = PassthroughProjector()
    model.config.vision_config.hidden_size = hidden_dim

    # freeze everything except projector (train) – LoRA comes next
    for n,p in model.named_parameters():
        p.requires_grad = ("projector" in n)

    # ---------- add LoRA ---------------------------------------------------
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    if rank0: model.print_trainable_parameters()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # ---------- data -------------------------------------------------------
    train_ds = CTChatDataset(args.csv, "train", get_train_transform())
    val_ds   = CTChatDataset(args.csv, "val",   get_val_transform())

    from torch.utils.data.distributed import DistributedSampler
    train_samp = DistributedSampler(train_ds) if distributed else None
    val_samp   = DistributedSampler(val_ds, shuffle=False) if distributed else None

    collator = BatchBuilder(processor)
    train_ld = DataLoader(train_ds, batch_size=args.bs, sampler=train_samp,
                          shuffle=(train_samp is None),
                          num_workers=args.workers, pin_memory=True,
                          collate_fn=collator)
    val_ld   = DataLoader(val_ds, batch_size=args.bs, sampler=val_samp,
                          shuffle=False, num_workers=args.workers,
                          pin_memory=True, collate_fn=collator)

    # ---------- optim ------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps = len(train_ld)*args.epochs
    sched = get_cosine_schedule_with_warmup(opt, 250, steps)

    # ---------- training loop ---------------------------------------------
    for ep in range(1, args.epochs+1):
        model.train()
        if distributed and hasattr(train_ld, "sampler"):
            train_ld.sampler.set_epoch(ep)

        bar = tqdm(train_ld, disable=not rank0)
        for batch in bar:
            batch = {k:(v.to(device) if torch.is_tensor(v) else v)
                     for k,v in batch.items()}
            loss = model(**batch).loss
            bar.set_description(f"Ep{ep} loss {loss.item():.4f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()

        # ---- simple val perplexity ---------------------------------------
        model.eval(); tot, cnt = 0.0, 0
        with torch.no_grad():
            for batch in val_ld:
                batch = {k:(v.to(device) if torch.is_tensor(v) else v)
                         for k,v in batch.items()}
                l = model(**batch).loss
                tot += l.item()*batch["labels"].size(0)
                cnt += batch["labels"].size(0)
        ppl = math.exp(tot/cnt)
        if rank0:
            logging.info(f"Epoch {ep}  val PPL = {ppl:.2f}")
            torch.save(model.state_dict(),
                       Path(args.out_dir)/f"stage2_ep{ep}.pt")

    # ---------- cleanup ----------------------------------------------------
    if distributed:
        dist.barrier(); dist.destroy_process_group()
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
