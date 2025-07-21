#!/usr/bin/env python3
"""
Stage ② – multimodal causal‑LM fine‑tuning
==========================================

• Loads MedGemma‑4B‑IT
• Replaces its SigLIP vision tower with:  CT_encoder  +  trained projector
• Inserts LoRA adapters in the Gemma LLM blocks
• Optimises   { projector weights  +  LoRA weights }   only
• Loss: next‑token (causal) on instruction‑following chat responses

Expected CSV columns:
    img_path      (path to .npz volume)
    instruction   (free‑text instruction / question)
    answer        (free‑text answer / report)
    split         (train / val)
"""

import os, argparse, logging, math, gc, torch, torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          AutoTokenizer, get_cosine_schedule_with_warmup,
                          set_seed)
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm

# -------- your local modules -------------------
from merlin import Merlin                           # 3‑D SigLIP encoder
from train_medgemma_stage1 import CTProjector            # same class as Stage ①
from training.ct_transform import get_train_transform, get_val_transform
from data_chat import CTChatDataset, chat_collate
# ------------------------------------------------


# ---------------------------------------------------------------------------
# 1.  A small wrapper so HF thinks this is a VisionTower
# ---------------------------------------------------------------------------
class CTVisionTower(nn.Module):
    """
    Mimics MedGemma's SiglipVisionTower API:
        - .config.hidden_size
        - forward(images) -> (B, 256, hidden)
    """
    def __init__(self, ct_encoder: nn.Module, projector: CTProjector,
                 hidden_dim: int):
        super().__init__()
        self.ct_encoder = ct_encoder
        self.projector  = projector
        self.config = type("cfg", (), {"hidden_size": hidden_dim})

    def forward(self, pixel_values, **kwargs):
        """
        pixel_values: torch.Tensor  (B, C, D, H, W)  prepared by dataset
        Returns     : BaseModelOutput with last_hidden_state (B, 256, hidden)
        """
        with torch.no_grad():
            feats = self.ct_encoder(pixel_values)          # (B, N, C) or (B,C)
        _, tokens = self.projector(feats)                  # (B, 256, H)
        # Gemma expects a BaseModelOutput with .last_hidden_state
        return BaseModelOutput(last_hidden_state=tokens)



# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stage1_ckpt", required=True, help="best_projector.pt")
    ap.add_argument("--ct_ckpt", required=True, help="Merlin weights")
    ap.add_argument("--model_id", default="google/medgemma-4b-it")
    ap.add_argument("--out_dir", default="stage2_lora_ckpt")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # ---------------- distributed -----------------------------------------
    distributed = "RANK" in os.environ
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank0  = not distributed or dist.get_rank() == 0

    if rank0:
        Path(args.out_dir).mkdir(exist_ok=True, parents=True)
        logging.basicConfig(level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(42 + local_rank)

    # ---------------- load CT encoder & projector -------------------------
    ct_encoder = Merlin(ImageEmbedding=True)
    ct_encoder.load_state_dict(torch.load(args.ct_ckpt, map_location="cpu"),
                               strict=False)
    ct_encoder.eval().requires_grad_(False).to(device)

    proj_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    projector = CTProjector(proj_ckpt["feat_dim"], proj_ckpt["hidden_dim"])
    projector.load_state_dict(proj_ckpt["projector"])
    projector.to(device)

    hidden_dim = proj_ckpt["hidden_dim"]

    # ---------------- load MedGemma (image‑text‑text) ---------------------
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                token=token,
                device_map={"": f"cuda:{local_rank}"},
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model_id, token=token)
    processor.tokenizer.padding_side = "right"

    # replace **both** references
    new_tower = CTVisionTower(ct_encoder, projector, hidden_dim)
    model.vision_tower = new_tower                  # convenience (not used)
    model.model.vision_tower = new_tower            # <- the one Gemma3Model calls
    model.config.vision_config.hidden_size = hidden_dim

    # ---- replace the multi‑modal projector ---------------------------
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

    model.model.multi_modal_projector = PassthroughProjector()

    # freeze everything except projector (already param.requires_grad=True)
    for n,p in model.named_parameters():
        p.requires_grad = ("projector" in n)

    # --------------- add LoRA to LLM side ---------------------------------
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # wrap with DDP (only LoRA & projector need gradients)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # ---------------- data ------------------------------------------------
    from torch.utils.data.distributed import DistributedSampler
    
    train_ds = CTChatDataset(args.csv, "train",
                             transform=get_train_transform(), three_ch=False)
    val_ds   = CTChatDataset(args.csv, "val",
                             transform=get_val_transform(),   three_ch=False)

    tr_sampler = DistributedSampler(train_ds) if distributed else None
    val_sampler= DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(train_ds, batch_size=args.bs, sampler=tr_sampler,
                              shuffle=(tr_sampler is None),
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=lambda b: chat_collate(b, processor))
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, sampler=val_sampler,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True,
                              collate_fn=lambda b: chat_collate(b, processor))

    # ---------------- optim ------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader)*args.epochs
    sched = get_cosine_schedule_with_warmup(opt, 250, total_steps)

    # ---------------- training loop ---------------------------------------
    for epoch in range(1, args.epochs+1):
        model.train()
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        bar = tqdm(train_loader, disable=not rank0)
        for batch in bar:
            batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / 1     # grad_accum=1 here
            bar.set_description(f"Ep{epoch} loss {loss.item():.4f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()

        # --- simple val perplexity (avg loss) -----------------------------
        model.eval(); tot, cnt = 0,0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                l = model(**batch).loss
                tot += l.item()*batch["labels"].size(0)
                cnt += batch["labels"].size(0)
        ppl = math.exp(tot/cnt)
        if rank0:
            logging.info(f"Epoch {epoch}  val PPL={ppl:.2f}")
            torch.save(model.state_dict(), Path(args.out_dir)/f"stage2_ep{epoch}.pt")

    # ---------------- cleanup ---------------------------------------------
    if distributed:
        dist.barrier(); dist.destroy_process_group()
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
