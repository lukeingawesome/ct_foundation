# --------------------------------------------------------------------
# Instruction-following chat dataset for CT volumes
# --------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch, pandas as pd, numpy as np
from torch.utils.data import Dataset

# create a clearly 3‑channel dummy (e.g. 8×8 black RGB)
DUMMY = Image.new("RGB", (8, 8))

# HU window utility function
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

@dataclass
class ChatSample:
    """Holds the (already loaded) CT volume and the chat messages list."""
    image: torch.Tensor           # (C, D, H, W)  – torch.float32
    messages: list                # list[dict] – Gemma chat template

class CTChatDataset(Dataset):
    """
    Loads rows (img_path, instruction, answer) and converts to ChatSample.
    """
    def __init__(self, csv, split, transform, three_ch=False):
        self.df = pd.read_csv(csv).query("split==@split").reset_index(drop=True)
        self.transform = transform
        self.three_ch = three_ch

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row.img_path)
        instr    = str(row.instruction)
        answer   = str(row.answer)

        # 1. read CT volume (same logic as Stage ①)
        with np.load(img_path) as npz:
            arr = npz["image"]                           # (C,D,H,W)
        if self.three_ch:
            arr = arr * 2500.0 - 1000.0                  # back to HU
            lung  = _hu_window_to_unit(arr[0], -600, 1000)
            medi  = _hu_window_to_unit(arr[0],   40,  400)
            bone  = _hu_window_to_unit(arr[0],  700, 1500)
            arr   = np.stack([lung, medi, bone], 0)      # (3,D,H,W)

        img = torch.from_numpy(arr).float()
        if self.transform: img = self.transform(img)     # (C,D,H,W)

        # 2. build chat messages (no <assistant/> tag for user prompt)
        messages = [
           {"role":"user",
            "content":[
               {"type":"text",  "text":"<task=report>"},
               {"type":"text",  "text":instr},
               {"type":"image"}]},
           {"role":"assistant",
            "content":[{"type":"text","text":answer}]}
        ]
        return ChatSample(image=img, messages=messages)


# ------------ collate --------------------------------------------------
def chat_collate(batch, processor):
    """
    Converts a list[ChatSample] → dict suitable for model(**batch)
    """
    images   = [s.image     for s in batch]      # real CT tensors (C,D,H,W)
    messages = [s.messages  for s in batch]

    texts = [processor.apply_chat_template(m, add_generation_prompt=False,
                                           tokenize=False).strip()
             for m in messages]

    # ---- keep exactly one <image> token per prompt -----------------
    enc = processor(
        text=texts,
        images=[[DUMMY]] * len(texts),   # list‑of‑lists → 1 dummy per sample
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # overwrite the dummy pixel_values with the real CT batch
    enc["pixel_values"] = torch.stack(images)    # (B,C,D,H,W)

    # build causal‑LM labels (mask pad & <image>)
    labels = enc["input_ids"].clone()
    special = [processor.tokenizer.pad_token_id,
               processor.image_token_id]
    for tok in special:
        labels[labels == tok] = -100
    enc["labels"] = labels
    return enc 