#!/usr/bin/env python3
# coding: utf-8
"""
Evaluate trained SigLIP model on chest-CT validation set
"""

from __future__ import annotations
import argparse, logging, os, sys, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Local imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "training"))
from training.ct_transform import get_val_transform
from merlin import Merlin

# ───────────────────────────────────────────────────────
def _hu_window_to_unit(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    """Clip a HU volume to the given window and scale to [0,1]."""
    lower, upper = center - width / 2.0, center + width / 2.0
    vol = np.clip(volume, lower, upper)
    return (vol - lower) / (upper - lower)

class CTDataset(Dataset):
    """Loads npz volumes and on-the-fly window mixing."""
    def __init__(self, csv: str, labels: List[str], split: str, transform=None, three_ch: bool = False):
        self.df = pd.read_csv(csv).query("split==@split").reset_index(drop=True)
        self.labels, self.transform, self.three_ch = labels, transform, three_ch
        logging.info(f"{split.capitalize():5s}: {len(self.df):,} vols")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            with np.load(row.img_path) as npz:
                arr = npz["image"]
        except Exception as e:
            logging.warning(f"Corrupt sample {row.img_path}: {e}")
            return self.__getitem__(0)  # fallback to first sample

        if self.three_ch:
            if arr.max() <= 1.0:
                arr = arr * 2500.0 - 1000.0

            if arr.ndim == 4:
                arr = arr[0]
            
            lung  = _hu_window_to_unit(arr,  -600, 1000)
            medi  = _hu_window_to_unit(arr,    40,  400)
            bone  = _hu_window_to_unit(arr,   700, 1500)
            arr = np.stack([lung, medi, bone], 0)

        img = torch.from_numpy(arr).float()
        if self.transform is not None: 
            img = self.transform(img)
        tgt = torch.from_numpy(row[self.labels].values.astype(np.float32))
        return img, tgt

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

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat)

# ───────────────────────────────────────────────────────
def _worker_init_fn(worker_id):
    """Worker initialization function to set single-threaded NumPy/BLAS."""
    np.random.seed()
    torch.set_num_threads(1)

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
def calculate_crg_score(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Calculate CRG (Clinically Relevant Grade) score."""
    if y_true.shape[0] == 0:
        return {"CRG": 0.0, "TP": 0, "FN": 0, "FP": 0, "X": 0, "A": 0, "r": 0.0, "U": 0.0, "score_s": 0.0}
    
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    
    num_images = y_true.shape[0]
    num_labels = len(labels)
    X = num_labels * num_images
    A = int(y_true.sum())
    
    if A == 0:
        return {"CRG": 0.0, "TP": TP, "FN": FN, "FP": FP, "X": X, "A": A, "r": 0.0, "U": 0.0, "score_s": 0.0}
    
    r = (X - A) / (2 * A)
    U = (X - A) / 2
    s = r * TP - r * FN - FP
    
    denominator = 2 * U - s
    crg = U / denominator if abs(denominator) > 1e-10 else 0.0
    crg = max(0.0, min(1.0, crg))
    
    return {
        "CRG": float(crg),
        "TP": TP,
        "FN": FN, 
        "FP": FP,
        "X": X,
        "A": A,
        "r": float(r),
        "U": float(U),
        "score_s": float(s)
    }

# ───────────────────────────────────────────────────────
def search_thresholds(logits: np.ndarray, gts: np.ndarray, step: float = 0.005) -> Tuple[np.ndarray, float]:
    """Search for optimal per-class thresholds."""
    best = np.zeros(logits.shape[1])
    for c in range(logits.shape[1]):
        t_candidates = np.arange(0.05, 0.95, step)
        f1s = [f1_score(gts[:, c], (logits[:, c] > t).astype(int), zero_division=0) for t in t_candidates]
        idx = int(np.argmax(f1s))
        best[c] = t_candidates[idx]
    macro = f1_score(gts, logits > best, average="macro", zero_division=0)
    return best, macro

# ───────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate(model, loader, device, labels, thresholds, loss_fn=None):
    """Evaluate model on validation set."""
    model.eval()
    logits, gts = [], []
    tot_loss = 0.0
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        out = model(x)
        if loss_fn is not None:
            tot_loss += loss_fn(out, y).item()
        logits.append(torch.sigmoid(out).cpu())
        gts.append(y.cpu())
    
    if logits:
        logits = torch.cat(logits, dim=0).numpy()
        gts = torch.cat(gts, dim=0).numpy()
    else:
        logits = np.array([]).reshape(0, len(labels))
        gts = np.array([]).reshape(0, len(labels))

    # Apply thresholds
    if isinstance(thresholds, float):
        preds = (logits > thresholds).astype(int)
    else:
        preds = (logits > thresholds[None, :]).astype(int)

    # Calculate CRG score
    crg_metrics = calculate_crg_score(gts, preds, labels)

    # Calculate macro AUC
    macro_auc = 0.0
    if gts.shape[0] > 0:
        try:
            macro_auc = roc_auc_score(gts, logits, average="macro")
        except ValueError as e:
            logging.warning(f"Could not calculate macro AUC: {e}")
            macro_auc = 0.5

    metrics = {
        "macro_f1": f1_score(gts, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(gts, preds, average="micro", zero_division=0),
        "macro_auc": macro_auc,
        "CRG": crg_metrics["CRG"]
    }
    
    # Per-class metrics
    pr, rc, f1, _ = precision_recall_fscore_support(
        gts, preds, average=None, zero_division=0, labels=range(len(labels)))
    
    auc = []
    for i in range(len(labels)):
        if gts.shape[0] > 0:
            try:
                unique_vals = np.unique(gts[:, i])
                if len(unique_vals) > 1:
                    auc_score = roc_auc_score(gts[:, i], logits[:, i])
                else:
                    auc_score = 0.5
                auc.append(auc_score)
            except ValueError as e:
                logging.warning(f"AUC calculation failed for {labels[i]}: {e}")
                auc.append(0.5)
        else:
            auc.append(0.0)
    
    for i, name in enumerate(labels):
        metrics[f"{name}_precision"] = pr[i] if i < len(pr) else 0.0
        metrics[f"{name}_recall"] = rc[i] if i < len(rc) else 0.0
        metrics[f"{name}_f1"] = f1[i] if i < len(f1) else 0.0
        metrics[f"{name}_auc"] = auc[i]

    return metrics, logits, gts

# ───────────────────────────────────────────────────────
def load_model_from_checkpoint(checkpoint_path: str, labels: List[str], device: torch.device):
    """Load model from checkpoint."""
    logging.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint (set weights_only=False for compatibility with numpy arrays)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Create the same model architecture as in training
    backbone = Merlin(ImageEmbedding=True)
    model = SigLIPClassifier(backbone, len(labels), drop=0.1)
    
    # Load the complete model state dict
    # Handle DDP prefix issue - checkpoint might have 'module.' prefix from distributed training
    state_dict = checkpoint["model"]
    
    # Check if state dict has 'module.' prefix (from DDP training)
    if any(key.startswith('module.') for key in state_dict.keys()):
        logging.info("Detected DDP checkpoint, removing 'module.' prefix...")
        # Remove 'module.' prefix from all keys
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info("Model loaded successfully with all weights")
    except Exception as e:
        logging.error(f"Failed to load state dict with strict=True: {e}")
        # Try with strict=False for debugging and print ALL missing keys
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            logging.warning(f"Loaded with strict=False. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            
            # Print ALL missing keys
            if missing_keys:
                logging.warning("ALL MISSING KEYS:")
                for i, key in enumerate(missing_keys):
                    logging.warning(f"  {i+1:3d}. {key}")
            
            # Print ALL unexpected keys  
            if unexpected_keys:
                logging.warning("ALL UNEXPECTED KEYS:")
                for i, key in enumerate(unexpected_keys):
                    logging.warning(f"  {i+1:3d}. {key}")
                    
        except Exception as e2:
            logging.error(f"Failed to load state dict even with strict=False: {e2}")
            raise RuntimeError(f"Could not load model weights: {e2}")
    
    model.to(device)
    model.eval()
    
    # Get thresholds if available
    thresholds = checkpoint.get("thresholds", np.full(len(labels), 0.5))
    
    logging.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    return model, thresholds

# ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate SigLIP model on validation set")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--csv", default="/data/all_ct_with_labels.csv", help="CSV file with data")
    parser.add_argument("--label-columns", nargs="+", default=None, help="Label columns")
    parser.add_argument("--three-channel", action="store_true", help="Use three-channel windowing")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--optimize-thresholds", action="store_true", help="Search for optimal thresholds")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--output-dir", default="eval_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load default labels if not provided
    if args.label_columns is None:
        labels_file = ROOT / "default_labels18.json"
        if labels_file.exists():
            args.label_columns = json.loads(labels_file.read_text())
            logging.info(f"Loaded {len(args.label_columns)} default labels")
        else:
            raise ValueError("No label columns provided and default_labels18.json not found")
    
    # Setup device - use CUDA if available, otherwise CPU
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        logging.info(f"Using device: {device} (CUDA available)")
    else:
        device = torch.device("cpu")
        if args.device.startswith("cuda"):
            logging.warning(f"CUDA requested but not available, falling back to CPU")
        else:
            logging.info(f"Using device: {device}")
    
    # Load model
    model, saved_thresholds = load_model_from_checkpoint(args.model_path, args.label_columns, device)
    
    # Create validation dataset with exact training parameters
    val_transform = get_val_transform()
    val_dataset = CTDataset(args.csv, args.label_columns, "val", val_transform, args.three_channel)
    
    # Match training DataLoader parameters exactly
    pin_memory_device = f"cuda:0" if torch.cuda.is_available() else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True,  # Match training: 4 workers
                           pin_memory_device=pin_memory_device,
                           persistent_workers=True, prefetch_factor=2, 
                           worker_init_fn=_worker_init_fn)  # Missing in evaluation!
    
    logging.info(f"Validation set: {len(val_dataset)} samples")
    
    # Create the same loss function used during training validation
    # Calculate pos_weight from training data (same as in training)
    train_df = pd.read_csv(args.csv).query("split=='train'")
    pos_w = torch.tensor(
        ((train_df.shape[0] - train_df[args.label_columns].sum(0).values)
         / (train_df[args.label_columns].sum(0).values + 1e-6)),
        dtype=torch.float, device=device)
    
    # Create focal loss with gamma=1.5 (from S2_loss_gamma15 checkpoint name)
    focal_gamma = 1.5  # Inferred from checkpoint path "S2_loss_gamma15"
    criterion = FocalBalancedLoss(pos_w, gamma=focal_gamma)
    
    # Evaluate with saved thresholds
    logging.info("Evaluating with saved thresholds...")
    metrics, logits, gts = evaluate(model, val_loader, device, args.label_columns, saved_thresholds, criterion)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS WITH SAVED THRESHOLDS")
    print("="*80)
    
    # Core metrics
    core_metrics = ["macro_f1", "micro_f1", "macro_auc", "CRG"]
    for metric in core_metrics:
        if metric in metrics:
            print(f"  {metric:>15s}: {metrics[metric]:.4f}")
    
    print("-"*40)
    print("Per-class F1 Scores:")
    for label in args.label_columns:
        f1_key = f"{label}_f1"
        if f1_key in metrics:
            print(f"  {label:>25s}: {metrics[f1_key]:.4f}")
    
    print("-"*40)
    print("Per-class AUC Scores:")
    for label in args.label_columns:
        auc_key = f"{label}_auc"
        if auc_key in metrics:
            print(f"  {label:>25s}: {metrics[auc_key]:.4f}")
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Results saved to {results_file}")
    
    # Optimize thresholds if requested
    if args.optimize_thresholds:
        logging.info("Searching for optimal thresholds...")
        optimal_thresholds, macro_f1_opt = search_thresholds(logits, gts)
        
        # Evaluate with optimal thresholds
        metrics_opt, _, _ = evaluate(model, val_loader, device, args.label_columns, optimal_thresholds, criterion)
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS WITH OPTIMIZED THRESHOLDS")
        print("="*80)
        
        for metric in core_metrics:
            if metric in metrics_opt:
                print(f"  {metric:>15s}: {metrics_opt[metric]:.4f}")
        
        print(f"\nOptimal thresholds:")
        for i, label in enumerate(args.label_columns):
            print(f"  {label:>25s}: {optimal_thresholds[i]:.3f}")
        
        # Save optimized results
        results_opt = {
            "metrics": metrics_opt,
            "thresholds": optimal_thresholds.tolist(),
            "macro_f1_with_optimal": macro_f1_opt
        }
        results_opt_file = output_dir / "evaluation_results_optimized.json"
        with open(results_opt_file, "w") as f:
            json.dump(results_opt, f, indent=2)
        logging.info(f"Optimized results saved to {results_opt_file}")
    
    print("="*80)

if __name__ == "__main__":
    main() 