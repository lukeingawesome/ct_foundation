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
import torch.distributed as dist
from pathlib import Path
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_cosine_schedule_with_warmup, 
    set_seed
)
from torch.optim import AdamW
from tqdm import tqdm

# ------------------------------------------------------------
#  YOUR own modules
#    * Merlin  ................. 3‑D SigLIP encoder (frozen)
#    * CustomCSVDataset ........ see user code (dataset_mode='ct')
#    * get_train_transform ..... from ct_transform.py
# ------------------------------------------------------------
from merlin import Merlin
from training.data import CustomCSVDataset, _hu_window_to_unit          # your file
from training.ct_transform import get_train_transform, get_val_transform      # your file
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
def build_dataloaders(csv_path: str, tokenizer, batch_size=8, workers=4, three_ch=False):
    tr_ds = CustomCSVDataset(
        csv_file=csv_path,
        transform=get_train_transform(),
        img_key="img_path", caption_key="findings",
        tokenizer=None, is_train=True,
        dataset_mode="ct", split="train", split_column="split",
        use_3channel=three_ch
    )
    val_ds = CustomCSVDataset(
        csv_file=csv_path,
        transform=get_val_transform(),
        img_key="img_path", caption_key="findings",
        tokenizer=None, is_train=False,
        dataset_mode="ct", split="val", split_column="split",
        use_3channel=three_ch
    )

    def collate(batch):
        imgs, caps = zip(*batch)
        imgs = torch.stack(imgs)
        tok = tokenizer(
            list(caps), padding=True, truncation=True,
            max_length=1024, return_tensors="pt"
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
    parser.add_argument("--three_ch", action="store_true")
    args = parser.parse_args()

    ###############################################
    # DDP initialisation
    distributed = "RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if distributed:
        torch.cuda.set_device(local_rank)      # unique GPU per rank
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}") if distributed else torch.device("cuda")
    rank0 = (not distributed) or dist.get_rank() == 0
    ###############################################

    if rank0:
        os.makedirs(args.out_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s")

    set_seed(42 + (dist.get_rank() if distributed else 0))

    # ---------- Load frozen CT encoder -------------------------------------
    ct_encoder = Merlin(ImageEmbedding=True)                  # your class
    if Path(args.ct_ckpt).is_file():
        sd = torch.load(args.ct_ckpt, map_location="cpu")
        ct_encoder.load_state_dict(sd, strict=False)
    ct_encoder.eval().requires_grad_(False).to(device)

    feat_dim = getattr(ct_encoder, "output_dim", 2048)

    # ---------- Load Gemma tokenizer / model ( text side is frozen) --------
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.gemma_id, token=token)

    gemma = AutoModelForCausalLM.from_pretrained(
        args.gemma_id,
        token=token,
        device_map={"": f"cuda:{local_rank}"},         # pin Gemma to this GPU
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    # Gemma3Config does not have `.hidden_size`.
    # Use the real embedding matrix instead (works for every model version).
    hidden_dim = gemma.get_input_embeddings().weight.shape[1]
    for p in gemma.parameters():
        p.requires_grad = False
    gemma.eval()

    # ---------- Projector to train -----------------------------------------
    projector = CTProjector(feat_dim, hidden_dim).to(device)
    if distributed:
        projector = torch.nn.parallel.DistributedDataParallel(
            projector,
            device_ids=[local_rank],            # explicit is safer
            output_device=local_rank
        )

    # ---------- Dataloaders ------------------------------------------------
    tr_ds = CustomCSVDataset(
        csv_file=args.csv,
        transform=get_train_transform(),
        img_key="img_path", caption_key="findings",
        tokenizer=None, is_train=True,
        dataset_mode="ct", split="train", split_column="split",
        use_3channel=args.three_ch
    )
    val_ds = CustomCSVDataset(
        csv_file=args.csv,
        transform=get_val_transform(),
        img_key="img_path", caption_key="findings",
        tokenizer=None, is_train=False,
        dataset_mode="ct", split="val", split_column="split",
        use_3channel=args.three_ch
    )

    def collate(batch):
        imgs, caps = zip(*batch)
        
        # Handle variable-sized tensors by finding max dimensions
        if len(imgs) > 0:
            # Get all shapes and find maximum dimensions
            shapes = [img.shape for img in imgs]
            max_shape = [max(dim) for dim in zip(*shapes)]
            
            # Pad all tensors to max_shape if needed
            padded_imgs = []
            for img in imgs:
                if img.shape != tuple(max_shape):
                    # Create padding for each dimension
                    padding = []
                    for i in range(len(img.shape)-1, -1, -1):  # reverse order for F.pad
                        pad_size = max_shape[i] - img.shape[i]
                        padding.extend([0, pad_size])
                    img = F.pad(img, padding, mode='constant', value=0)
                padded_imgs.append(img)
            imgs = torch.stack(padded_imgs)
        else:
            imgs = torch.stack(imgs)
            
        tok = tokenizer(
            list(caps), padding=True, truncation=True,
            max_length=1024, return_tensors="pt"
        )
        return imgs, tok["input_ids"], tok["attention_mask"]

    from torch.utils.data.distributed import DistributedSampler
    tr_sampler = DistributedSampler(tr_ds) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    tr_loader = DataLoader(
        tr_ds, batch_size=args.bs, shuffle=(tr_sampler is None),
        sampler=tr_sampler, num_workers=args.workers, pin_memory=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        sampler=val_sampler, num_workers=args.workers, pin_memory=True, collate_fn=collate
    )

    # ---------- Optim & sched ---------------------------------------------
    opt = AdamW(projector.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = len(tr_loader) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, 250, total_steps)

    # ---------- Training loop ---------------------------------------------
    best_val = -1.0
    temperature = 0.07

    def encode_text(input_ids, attn_mask):
        with torch.no_grad():
            out = gemma.get_input_embeddings()(input_ids.to(device))  # (B, L, H)
            txt_emb = mean_pool_hidden(out, attn_mask.to(device))
            return F.normalize(txt_emb, dim=-1)

    for epoch in range(1, args.epochs + 1):
        if distributed: 
            tr_loader.sampler.set_epoch(epoch)
        projector.train()
        total_loss = 0.0
        
        # Training progress bar (only on rank 0)
        train_bar = tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs}", 
                        disable=not rank0, leave=False)
        
        for imgs, ids, mask in train_bar:
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
            
            # Update progress bar with current loss
            if rank0:
                train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg = total_loss / len(tr_loader)
        if rank0:
            logging.info(f"Epoch {epoch:02d}  train_loss={avg:.4f}")

        # ---------- quick val retrieval recall@1 -----------------------
        projector.eval()
        sims, correct = 0, 0
        
        # Validation progress bar (only on rank 0)
        val_bar = tqdm(val_loader, desc="Validation", 
                      disable=not rank0, leave=False)
        
        with torch.no_grad():
            for imgs, ids, mask in val_bar:
                imgs = imgs.to(device); feats = ct_encoder(imgs)
                img_emb, _ = projector(feats)
                txt_emb = encode_text(ids, mask)
                sim = (img_emb @ txt_emb.t()).diag()          # (B,)
                sims += sim.numel()
                correct += (sim.argmax(dim=0) == 0).sum().item()   # recall@1

        if distributed:
            total_corr = torch.tensor(correct, device=device)
            total_sims = torch.tensor(sims,    device=device)
            dist.all_reduce(total_corr, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_sims, op=dist.ReduceOp.SUM)
            r1 = (total_corr / total_sims).item()
        else:
            r1 = correct / sims
        if rank0:
            logging.info(f"           val_recall@1 = {r1:.3%}")

        # ---------- save best ------------------------------------------
        if rank0 and r1 > best_val:
            best_val = r1
            model_state = projector.module.state_dict() if distributed else projector.state_dict()
            torch.save({"projector": model_state,
                        "epoch": epoch,
                        "feat_dim": feat_dim,
                        "hidden_dim": hidden_dim},
                       Path(args.out_dir) / "best_projector.pt")
            logging.info("           ✓  saved new best")

    # ---------------------------------------------------------------------
    if rank0:
        logging.info(f"Done.  Best val recall@1 = {best_val:.3%}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
