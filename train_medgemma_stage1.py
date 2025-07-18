#!/usr/bin/env python3
"""
Stage ① – Pre‑train a projector that maps frozen 3‑D CT encoder features
          → Gemma/MedGemma token‑embedding space (hidden_size).

Loss      : CLIP‑style InfoNCE on (image ↔ caption) pairs.
Frozen    : CT encoder, Gemma tokenizer + embedding matrix, all LLM blocks.
Trainable : Projector only (an MLP that outputs 256 visual tokens).

Data      : CSV with columns  • img_path  • findings  • split (train/val).
Transforms: re‑use get_train_transform / get_val_transform from your code.
"""

import argparse, os, math, random, json, logging, torch, gc
from pathlib import Path
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_cosine_schedule_with_warmup, 
    AdamW, set_seed
)
os.environ["HF_HOME"] = "/model/huggingface"

# ------------------------------------------------------------
#  YOUR own modules
#    * Merlin  ................. 3‑D SigLIP encoder (frozen)
#    * CustomCSVDataset ........ see user code (dataset_mode='ct')
#    * get_train_transform ..... from ct_transform.py
# ------------------------------------------------------------
from merlin import Merlin
from dataloader import CustomCSVDataset, _hu_window_to_unit          # your file
from ct_transform import get_train_transform, get_val_transform      # your file
# ------------------------------------------------------------

def mean_pool_hidden(hidden, attention_mask):
    """Mean‑pool the last hidden states, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).expand_as(hidden)            # (B,L,H)
    masked = hidden * mask
    return masked.sum(1) / mask.sum(1).clamp(min=1e-9)               # (B,H)

class CTProjector(nn.Module):
    """
    • Takes CT encoder patch tokens (B, C, D, H, W)  → flatten → (B, N, feat_dim)
    • Projects to Gemma hidden size and reduces to 256 visual tokens (B, 256, H).
    """
    def __init__(self, feat_dim: int, hidden_dim: int, num_tokens: int = 256):
        super().__init__()
        self.reduce = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.num_tokens = num_tokens
        # Learned positional encodings (optional but useful)
        self.pos = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

    def forward(self, x):
        """
        x : (B, feat_dim)      – if your encoder already returns global feat
            OR
            (B, N, feat_dim)   – patch tokens
        Output:
            (B, hidden_dim)    – global embedding for contrastive loss
            (B, num_tokens, hidden_dim) – full token sequence for later stages
        """
        if x.ndim == 2:                      # (B, feat_dim)
            x = x.unsqueeze(1)               # (B, 1, C)

        # If too many patch tokens, uniform subsample to num_tokens
        B, N, C = x.shape
        if N > self.num_tokens:
            idx = torch.linspace(0, N-1, self.num_tokens, device=x.device).long()
            x = x[:, idx]                    # (B, 256, C)
        elif N < self.num_tokens:
            # Repeat / interpolate to reach 256
            repeat = math.ceil(self.num_tokens / N)
            x = x.repeat(1, repeat, 1)[:, :self.num_tokens]

        x = self.reduce(x) + self.pos        # (B, 256, hidden)
        # Global CLS‑style embedding = mean‑pool
        img_emb = x.mean(1)                  # (B, hidden)
        img_emb = F.normalize(img_emb, dim=-1)
        return img_emb, x                    # (for Stage ②)

# -------------------------------------------------------------------------
def build_dataloaders(csv_path: str, tokenizer, batch_size=8, workers=4):
    tr_ds = CustomCSVDataset(
        csv_file=csv_path,
        transform=get_train_transform(),
        img_key="img_path", caption_key="findings",
        tokenizer=None, is_train=True,
        dataset_mode="ct", split="train", split_column="split",
        use_3channel=False
    )
    val_ds = CustomCSVDataset(
        csv_file=csv_path,
        transform=get_val_transform(),
        img_key="img_path", caption_key="findings",
        tokenizer=None, is_train=False,
        dataset_mode="ct", split="val", split_column="split",
        use_3channel=False
    )

    def collate(batch):
        imgs, caps = zip(*batch)
        imgs = torch.stack(imgs)
        tok = tokenizer(
            list(caps), padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        return imgs, tok["input_ids"], tok["attention_mask"]

    tr_loader = DataLoader(tr_ds, batch_size, shuffle=True,
                           num_workers=workers, pin_memory=True,
                           collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True,
                            collate_fn=collate)
    return tr_loader, val_loader

# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV w/ img_path, findings, split")
    parser.add_argument("--ct_ckpt", default="/model/1c_siglip2/pytorch_model.bin")
    parser.add_argument("--gemma_id", default="google/medgemma-4b-it")
    parser.add_argument("--out_dir", default="projector_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load frozen CT encoder -------------------------------------
    ct_encoder = Merlin(ImageEmbedding=True)                  # your class
    if Path(args.ct_ckpt).is_file():
        sd = torch.load(args.ct_ckpt, map_location="cpu")
        ct_encoder.load_state_dict(sd, strict=False)
    ct_encoder.eval().requires_grad_(False).to(device)

    feat_dim = getattr(ct_encoder, "output_dim", 2048)

    # ---------- Load Gemma tokenizer / model ( text side is frozen) --------
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_id)
    gemma = AutoModelForCausalLM.from_pretrained(
        args.gemma_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    hidden_dim = gemma.config.hidden_size
    for p in gemma.parameters():
        p.requires_grad = False
    gemma.eval()

    # ---------- Projector to train -----------------------------------------
    projector = CTProjector(feat_dim, hidden_dim).to(device)

    # ---------- Dataloaders ------------------------------------------------
    tr_loader, val_loader = build_dataloaders(args.csv, tokenizer,
                                              batch_size=args.bs, workers=args.workers)

    # ---------- Optim & sched ---------------------------------------------
    opt = AdamW(projector.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = len(tr_loader) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, 250, total_steps)

    # ---------- Training loop ---------------------------------------------
    best_val = -1.0
    temperature = 0.07

    def encode_text(input_ids, attn_mask):
        with torch.no_grad():
            out = gemma.model.embed_tokens(input_ids.to(device))       # (B,L,H)
            txt_emb = mean_pool_hidden(out, attn_mask.to(device))
            return F.normalize(txt_emb, dim=-1)

    for epoch in range(1, args.epochs + 1):
        projector.train()
        total_loss = 0.0
        for imgs, ids, mask in tr_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                feats = ct_encoder(imgs)                 # (B, feat_dim) or (B,N,C)
            img_emb, _ = projector(feats)                # (B, H)

            txt_emb = encode_text(ids, mask)             # (B, H)

            # InfoNCE ----------------------------------------------------
            logits = img_emb @ txt_emb.t() / temperature       # (B,B)
            labels = torch.arange(img_emb.size(0), device=device)
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.t(), labels)
            loss = (loss_i + loss_t) / 2

            opt.zero_grad()
            loss.backward()
            opt.step(); sched.step()

            total_loss += loss.item()

        avg = total_loss / len(tr_loader)
        logging.info(f"Epoch {epoch:02d}  train_loss={avg:.4f}")

        # ---------- quick val retrieval recall@1 -----------------------
        projector.eval()
        sims, correct = 0, 0
        with torch.no_grad():
            for imgs, ids, mask in val_loader:
                imgs = imgs.to(device); feats = ct_encoder(imgs)
                img_emb, _ = projector(feats)
                txt_emb = encode_text(ids, mask)
                sim = (img_emb @ txt_emb.t()).diag()          # (B,)
                sims += sim.numel()
                correct += (sim.argmax(dim=0) == 0).sum().item()   # recall@1

        r1 = correct / sims
        logging.info(f"           val_recall@1 = {r1:.3%}")

        # ---------- save best ------------------------------------------
        if r1 > best_val:
            best_val = r1
            torch.save({"projector": projector.state_dict(),
                        "epoch": epoch,
                        "feat_dim": feat_dim,
                        "hidden_dim": hidden_dim},
                       Path(args.out_dir) / "best_projector.pt")
            logging.info("           ✓  saved new best")

    # ---------------------------------------------------------------------
    logging.info(f"Done.  Best val recall@1 = {best_val:.3%}")
    gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
