#!/usr/bin/env python3
# coding: utf-8
"""
Finetune SigLIP on chest‑CT multilabel classification (18 findings)

Adds:
• 3‑D aug, discriminative LR, focal‑BCE combo, dynamic threshold search
• Gradual unfreezing, SWA, EMA, WandB histograms
• Cleaner CLI, better typing, fewer global variables
• Multi-GPU distributed training support
"""

from __future__ import annotations
import argparse, logging, os, random, sys, warnings, re, math, json
import torch.distributed as dist
from pathlib import Path
from typing import Dict, List, Tuple, Optional
# test
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# --- make PyTorch checkpoint use the non‑reentrant engine -------------
import torch.utils.checkpoint as cp
if hasattr(cp, "config"):
    cp.config.set_reentrant(False)        # pytorch ≥1.13
# ----------------------------------------------------------------------

# ───────────────────────────────────────────────────────
try:
    import wandb; WANDB_OK = True
except ImportError:
    WANDB_OK = False; warnings.warn("wandb unavailable – external logging disabled")

# local imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "training"))
from training.ct_transform import get_train_transform, get_val_transform
from merlin import Merlin

# ───────────────────────────────────────────────────────
def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

# ───────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True
    # speed up matmuls on Ampere/Hopper
    torch.set_float32_matmul_precision('high')

# ───────────────────────────────────────────────────────
class FocalBalancedLoss(nn.Module):
    """Focal loss + BCE pos‑weight (works better for extreme imbalance)."""
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__(); self.w = pos_weight; self.g = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                 reduction="none", pos_weight=self.w)
        p_t = torch.exp(-bce)
        loss = (1 - p_t) ** self.g * bce
        return loss.mean()

# ───────────────────────────────────────────────────────
class SigLIPClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, drop: float = .1):
        super().__init__()
        self.backbone = backbone
        dim = getattr(backbone, "output_dim", 2048)
        self.head = nn.Sequential(
            nn.Dropout(drop), nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(dim // 2, n_classes)
        )
        self.head.apply(self._init)

    @staticmethod
    def _init(m):                                 # Xavier normal
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):                         # (B,C,D,H,W)
        feat = self.backbone(x).flatten(1)
        return self.head(feat)

# ───────────────────────────────────────────────────────
def _hu_window_to_unit(volume: np.ndarray,
                       center: float,
                       width: float) -> np.ndarray:
    """
    Clip a HU volume to the given window and scale to [0,1].
    Args
        volume : raw HU ndarray, shape (D,H,W) or (C,D,H,W)
        center : window centre in HU
        width  : window width in HU
    """
    lower, upper = center - width / 2.0, center + width / 2.0
    vol = np.clip(volume, lower, upper)
    return (vol - lower) / (upper - lower)

class CTDataset(Dataset):
    """Loads npz volumes and on‑the‑fly window mixing."""
    def __init__(self, csv: str, labels: List[str], split: str,
                 transform=None, three_ch: bool = True):
        self.df = pd.read_csv(csv).query("split==@split").reset_index(drop=True)
        self.labels, self.transform, self.three_ch = labels, transform, three_ch
        logging.info(f"{split.capitalize():5s}: {len(self.df):,} vols")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            with np.load(row.img_path) as npz:
                arr = npz["image"]                         # (C,D,H,W) or (1,D,H,W)
        except Exception as e:
            logging.warning(f"Corrupt sample {row.img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.three_ch:
            if arr.max() <= 1.0:                         # heuristic
                arr = arr * 2500.0 - 1000.0              # back‑to‑HU

            # arr shape: (C,D,H,W) or (D,H,W); unify to (D,H,W)
            if arr.ndim == 4:
                arr = arr[0]        # assume first channel is full‑range HU
            
            # Generate three standard windows
            #    lung (centre -600, width 1000)
            #    mediastinum (centre  40, width  400)
            #    bone (centre 700, width 1500)
            lung  = _hu_window_to_unit(arr,  -600, 1000)
            medi  = _hu_window_to_unit(arr,    40,  400)
            bone  = _hu_window_to_unit(arr,   700, 1500)

            channels = [lung, medi, bone]
            
            # optional random channel masking / swapping
            if random.random() < .15: random.shuffle(channels)
            if random.random() < .15:
                i = random.randint(0, 2); channels[i] = np.zeros_like(channels[i])
            arr = np.stack(channels, 0)                    # (3,D,H,W)

        img = torch.from_numpy(arr).float()
        if self.transform is not None: img = self.transform(img)
        tgt = torch.from_numpy(row[self.labels].values.astype(np.float32))
        return img, tgt

# ───────────────────────────────────────────────────────
def make_loaders(csv: str, labels: List[str], bs: int, nw: int, three_ch: bool,
                 balance: bool, rank: int = 0, world_size: int = 1):
    train_ds = CTDataset(csv, labels, "train", get_train_transform(), three_ch)
    val_ds   = CTDataset(csv, labels, "val",   get_val_transform(),   three_ch)

    # Distributed training setup
    use_distributed = world_size > 1
    
    if use_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(train_ds, bs, sampler=train_sampler,
                                  num_workers=nw, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, bs, sampler=val_sampler,
                                num_workers=nw, pin_memory=True, persistent_workers=True)
    elif balance:
        y = train_ds.df[labels].values
        # sample weights = #neg / #pos  (per class)  → flatten for multilabel
        cls_weights = (y.shape[0] - y.sum(0)) / (y.sum(0) + 1e-6)
        w = (y * cls_weights).sum(1)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        train_loader = DataLoader(train_ds, bs, sampler=sampler,
                                  num_workers=nw, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, bs, shuffle=False,
                                num_workers=nw, pin_memory=True, persistent_workers=True)
    else:
        train_loader = DataLoader(train_ds, bs, shuffle=True,
                                  num_workers=nw, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, bs, shuffle=False,
                                num_workers=nw, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader

# ───────────────────────────────────────────────────────
def load_backbone(ckpt: str) -> Merlin:
    model = Merlin(ImageEmbedding=True)
    if ckpt and Path(ckpt).is_file():
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict({k[7:]: v for k, v in sd.items()
                               if k.startswith("visual.")}, strict=False)
        logging.info(f"Loaded SigLIP visual weights ↗ {ckpt}")
    else:
        logging.warning("No pre‑trained weights found – backbone randomly initialised")
    return model

# ───────────────────────────────────────────────────────
def freeze_regex(model: nn.Module, patterns: List[str]):
    """Set requires_grad = False on params matching *any* of the regex patterns."""
    if not patterns: return
    compiled = [re.compile(p) for p in patterns]
    for n, p in model.named_parameters():
        if any(r.search(n) for r in compiled):
            p.requires_grad = False

# ───────────────────────────────────────────────────────
@torch.inference_mode()
def search_thresholds(logits: np.ndarray, gts: np.ndarray,
                      step: float = 0.005) -> Tuple[np.ndarray, float]:
    best = np.zeros(logits.shape[1]); macro = 0.
    for c in range(logits.shape[1]):
        t_candidates = np.arange(0.05, 0.95, step)
        f1s = [f1_score(gts[:, c], (logits[:, c] > t).astype(int),
                        zero_division=0) for t in t_candidates]
        idx = int(np.argmax(f1s)); best[c] = t_candidates[idx]
    macro = f1_score(gts, logits > best, average="macro", zero_division=0)
    return best, macro

# ───────────────────────────────────────────────────────
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes in distributed training."""
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all processes."""
    if not dist.is_initialized():
        return tensor
    
    # Get the size of each tensor on each process
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    return torch.cat(tensor_list, dim=0)

# ───────────────────────────────────────────────────────
def evaluate(model, loader, loss_fn, device, labels,
             thresholds: np.ndarray | float):
    model.eval(); tot_loss, logits, gts = 0., [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            out = model(x); tot_loss += loss_fn(out, y).item()
            logits.append(torch.sigmoid(out).cpu()); gts.append(y.cpu())
    
    if logits:
        logits = torch.cat(logits, dim=0).numpy()
        gts = torch.cat(gts, dim=0).numpy()
    else:
        logits = np.array([]).reshape(0, len(labels))
        gts = np.array([]).reshape(0, len(labels))

    if isinstance(thresholds, float):
        preds = (logits > thresholds).astype(int)
    else:
        preds = (logits > thresholds[None, :]).astype(int)

    metrics = {
        "val_loss":  tot_loss / len(loader) if len(loader) > 0 else 0.0,
        "macro_f1":  f1_score(gts, preds, average="macro", zero_division=0),
        "micro_f1":  f1_score(gts, preds, average="micro", zero_division=0),
        "macro_auc": roc_auc_score(gts, logits, average="macro") if gts.shape[0] > 0 else 0.0
    }
    # per‑class metrics (precision / recall / auc)
    pr, rc, f1, _ = precision_recall_fscore_support(
        gts, preds, average=None, zero_division=0, labels=range(len(labels)))
    auc = [roc_auc_score(gts[:, i], logits[:, i]) if gts.shape[0] > 0 else 0.0 
           for i in range(len(labels))]
    for i, name in enumerate(labels):
        metrics[f"{name}_precision"] = pr[i] if i < len(pr) else 0.0
        metrics[f"{name}_recall"]    = rc[i] if i < len(rc) else 0.0
        metrics[f"{name}_f1"]        = f1[i] if i < len(f1) else 0.0
        metrics[f"{name}_auc"]       = auc[i]

    return metrics, logits, gts

# ───────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    # ───────── DDP
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)),
                        help="passed by torchrun; no need to set manually")
    # data / labels
    parser.add_argument("--csv", default="/data/all_ct_with_labels.csv")
    parser.add_argument("--label-columns", nargs="+", default=None)
    parser.add_argument("--three-channel", action="store_true")
    parser.add_argument("--balance-sampler", action="store_true")
    # model
    parser.add_argument("--pretrained", default="/model/1c_siglip/pytorch_model.bin")
    parser.add_argument("--freeze-regex", default="")
    parser.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-backbone-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "warmup"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Keep backbone frozen for N epochs")
    # regularisation
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--use-swa", action="store_true", default=True)
    # ───────── WandB (new)
    parser.add_argument("--wandb-project", default="siglip-ct",
                        help="Weights‑and‑Biases *project* (entity/project)")
    parser.add_argument("--wandb-name", default=None,
                        help="Run name as shown in the WandB UI "
                             "(defaults to output‑folder name)")
    # misc
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args(argv)

    # world size / rank
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank       = int(os.getenv("RANK", 0))
    distributed = world_size > 1

    # ensure only rank‑0 prints unless explicitly desired
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()) if rank == 0 else logging.ERROR,
        format=f"%(asctime)s | r{rank} | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # initialise DDP
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    
    set_seed(42 + rank)  # Different seed per process
    
    if rank == 0:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # default labels if not supplied
    if args.label_columns is None:
        args.label_columns = json.loads(Path(ROOT/"default_labels18.json").read_text())
        if rank == 0:
            logging.info(f"Loaded {len(args.label_columns)} default labels")

    # ───────── initialise WandB (after args, before training loop)
    if WANDB_OK and rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or Path(args.output).name,
            config=vars(args),
            dir=args.output,          # run files go next to checkpoints
            reinit=False,             # safer when ↻ in same process
        )
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # data
    num_workers = max(2, (os.cpu_count() or 4) // 2)
    tr_loader, val_loader = make_loaders(
        args.csv, args.label_columns, args.batch_size, num_workers,
        args.three_channel, args.balance_sampler)

    # wrap samplers in DistributedSampler
    if distributed:
        if args.balance_sampler:
            warnings.warn("Balance‑sampler disabled in DDP (fallback to uniform shuffle)")
        tr_loader = DataLoader(
            tr_loader.dataset,
            batch_size=args.batch_size,
            sampler=DistributedSampler(tr_loader.dataset, shuffle=True),
            num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=args.batch_size,
            sampler=DistributedSampler(val_loader.dataset, shuffle=False),
            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # model
    backbone = load_backbone(args.pretrained)
    model = SigLIPClassifier(backbone, len(args.label_columns), args.dropout).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True, static_graph=True)  # Required when using frozen parameters

    # Get the actual model for parameter access (unwrap DDP if needed)
    model_module = model.module if distributed else model

    freeze_regex(model_module.backbone, [s for s in args.freeze_regex.split(",") if s])
    # param groups: backbone vs head
    param_groups = [
        {"params": [p for n, p in model_module.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": args.lr * args.lr_backbone_mult},
        {"params": [p for n, p in model_module.named_parameters()
                    if "backbone" not in n and p.requires_grad],
         "lr": args.lr}
    ]
    if args.optimizer == "adamw":
        opt = AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        opt = SGD(param_groups, weight_decay=args.weight_decay, momentum=0.9)

    scaler = GradScaler(enabled=args.amp)

    # scheduler
    total_steps = math.ceil(len(tr_loader) / args.grad_accum) * args.epochs
    if args.scheduler == "warmup":
        sch = get_cosine_schedule_with_warmup(opt, args.warmup_steps, total_steps)
    elif args.scheduler == "cosine":
        sch = CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sch = ReduceLROnPlateau(opt, mode="max", factor=.5, patience=5)

    # SWA + EMA
    swa_model = AveragedModel(model_module) if args.use_swa else None
    swa_start = int(args.epochs * 0.6)
    swa_scheduler = SWALR(opt, swa_lr=args.lr * 0.05) if args.use_swa else None
    ema = AveragedModel(model_module, avg_fn=None) if args.use_ema else None

    # loss
    pos_w = torch.tensor(
        ( (tr_loader.dataset.df[args.label_columns].values.shape[0]
            - tr_loader.dataset.df[args.label_columns].sum(0).values)
          / (tr_loader.dataset.df[args.label_columns].sum(0).values + 1e-6)),
        dtype=torch.float, device=device)
    criterion = FocalBalancedLoss(pos_w)

    best_f1, patience = 0., 0
    thresholds = np.full(len(args.label_columns), args.threshold, dtype=np.float32)

    if WANDB_OK and rank == 0:
        wandb.watch(model, log='all', log_freq=100)

    # training loop ────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        if distributed and isinstance(tr_loader.sampler, DistributedSampler):
            tr_loader.sampler.set_epoch(epoch)
        
        # (1) freeze schedule
        if epoch == args.freeze_epochs + 1:
            model_module = model.module if distributed else model
            for p in model_module.backbone.parameters(): p.requires_grad = True
            if rank == 0:
                logging.info("Backbone unfrozen")

        model.train(); running = 0.
        
        # Only show progress bar on main process
        if rank == 0:
            pbar = tqdm(enumerate(tr_loader, 1), total=len(tr_loader), ncols=120,
                        desc=f"epoch[{epoch:03d}]")
        else:
            pbar = enumerate(tr_loader, 1)
        
        opt.zero_grad()

        for step, (x, y) in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with autocast(enabled=args.amp):
                loss = criterion(model(x), y) / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum

            if step % args.grad_accum == 0 or step == len(tr_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                if ema: ema.update_parameters(model.module if distributed else model)
                if swa_model and epoch >= swa_start: swa_model.update_parameters(model.module if distributed else model)
            
            if rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=f"{running/step:.4f}")

        # scheduler step
        if args.scheduler == "plateau":
            # will step later after val_metrics computed
            pass
        elif args.scheduler == "warmup":
            sch.step()
        else:
            sch.step()

        # (2) validation — only on rank‑0
        if rank == 0:
            eval_model = ema if ema else model.module if distributed else model
            val_metrics, val_logits, val_gts = evaluate(
                eval_model, val_loader, criterion, device,
                args.label_columns, thresholds)
        else:
            val_metrics = {}

        # dynamic threshold search each epoch (only on main process)
        if rank == 0:
            thresholds, macro_at_opt = search_thresholds(val_logits, val_gts)
            val_metrics["macro_f1_opt"] = macro_at_opt

        # (3) SWA scheduler step
        if swa_model and epoch >= swa_start and swa_scheduler:
            swa_scheduler.step()

        if rank == 0 and args.scheduler == "plateau":
            sch.step(val_metrics["macro_f1"])

        if rank == 0:
            msg = " | ".join(f"{k}:{v:.4f}" if isinstance(v, float) else f"{k}:{v}"
                             for k, v in val_metrics.items() if not k.endswith("_precision")
                                                                        and not k.endswith("_recall"))
            logging.info(f"Epoch {epoch:03d} – {msg}")

        if WANDB_OK and rank == 0:
            wandb.log({"epoch": epoch, **val_metrics})

        if rank == 0:
            ckpt_path = Path(args.output) / f"epoch_{epoch}.pth"
            torch.save({"epoch": epoch, "model": eval_model.state_dict(),
                        "opt": opt.state_dict(), "sch": sch.state_dict(),
                        "metrics": val_metrics, "thresholds": thresholds}, ckpt_path)

            if val_metrics["macro_f1"] > best_f1:
                best_f1, patience = val_metrics["macro_f1"], 0
                best_path = Path(args.output) / "best.pth"
                if best_path.exists():
                    best_path.unlink()
                best_path.symlink_to(ckpt_path.name, target_is_directory=False)
            else:
                patience += 1
                if patience >= args.patience:
                    logging.info("Early stopping ↑ patience exhausted")
                    break

    # (6) SWA batchnorm update + final eval
    if rank == 0 and swa_model:
        torch.optim.swa_utils.update_bn(tr_loader, swa_model, device=device)
        final_model = swa_model
    else:
        final_model = ema if ema else (model.module if distributed else model)

    if rank == 0:
        final_metrics, *_ = evaluate(final_model, val_loader,
                                     criterion, device, args.label_columns, thresholds)
        logging.info("Final metrics:\n" +
                     "\n".join(f"  {k:>15s}: {v:.4f}" for k, v in final_metrics.items()
                               if isinstance(v, float)))
        if WANDB_OK:
            wandb.summary.update(final_metrics)
            wandb.finish()

    # clean up
    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__": main()
