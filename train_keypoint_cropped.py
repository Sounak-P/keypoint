"""
Improved Keypoint Training Script — Crop-Aware + SE-UNet + Focal-Dice Loss
===========================================================================
Key improvements over the original kaggle_cvat_keypoint_train_infer.py:

  1. CROP-AWARE DATASET
     Uses the COCO bbox field to crop each annotated region (+ padding),
     then remaps keypoints into crop-space.  This exactly mirrors what
     infer_two_stage_video.py does at inference time, eliminating the
     train/inference distribution gap that caused bad detection at distance.

  2. STRONG AUGMENTATION (Albumentations)
     HorizontalFlip, ShiftScaleRotate, Perspective, ColorJitter,
     MotionBlur, CoarseDropout — all with correct keypoint mirroring.
     Falls back to basic torchvision transforms if albumentations missing.

  3. BETTER LOSS: FocalHeatmapLoss = FocalMSE + DiceLoss
     Heavily suppresses easy-background gradients, forces the network
     to produce sharp confident peaks rather than diffuse blobs.

  4. SE-UNet (Squeeze-Excite attention)
     Channel-wise attention after each encoder block.  Same output shape
     as the original UNet — the trained checkpoint is a drop-in replacement
     for final_model.ckpt used by infer_two_stage_video.py.

  5. COSINE ANNEALING LR + LINEAR WARMUP
     Better convergence than ReduceLROnPlateau with fixed patience.

Usage (local):
  python train_keypoint_cropped.py

On Kaggle change DATA_JSON / IMAGE_DIR / BASE_DIR at the top of Config.
"""

import os
import sys
import json
import math
import time
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as TF
import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")

# ── Albumentations (optional) ────────────────────────────────────────────────
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("WARN: albumentations not found — using basic transforms. "
          "Run: pip install albumentations")


# ================================================================================
# Config
# ================================================================================
class Config:
    # ── Data paths ──────────────────────────────────────────────────────────────
    if os.path.exists("/kaggle"):
        BASE_DIR  = "/kaggle/working"
        DATA_JSON = "/kaggle/input/datasets/sounakp/ketpoints/keypointframes/annotations/person_keypoints_default.json"
        IMAGE_DIR = "/kaggle/input/datasets/sounakp/ketpoints/keypointframes/images"
    else:
        _script = Path(__file__).parent
        BASE_DIR  = str(_script)
        DATA_JSON = str(_script / "keypointframes" / "annotations" / "person_keypoints_default.json")
        IMAGE_DIR = str(_script / "keypointframes" / "images")

    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_cropped")
    VIS_DIR    = os.path.join(OUTPUT_DIR, "visualizations")

    # ── Model ───────────────────────────────────────────────────────────────────
    INPUT_SIZE    = 384          # larger than original 256 → more detail in crop
    NUM_KEYPOINTS = 4
    BASE_CHANNELS = 64           # same as original — checkpoint compatible

    # ── Heatmap ─────────────────────────────────────────────────────────────────
    HEATMAP_SIGMA = 8            # larger σ → easier gradients; focal loss sharpens peaks

    # ── Training ────────────────────────────────────────────────────────────────
    MAX_EPOCHS         = 100
    BATCH_SIZE         = 8       # reduce to 4 if OOM on CPU
    LEARNING_RATE      = 2e-4
    WEIGHT_DECAY       = 1e-4
    NUM_WORKERS        = 0       # set >0 if on Linux
    SEED               = 42
    TRAIN_VAL_SPLIT    = 0.85    # 85% train, 15% val

    # ── Crop ────────────────────────────────────────────────────────────────────
    CROP_PAD_RATIO = 0.15        # matches infer_two_stage_video.py default
    MIN_CROP_SIZE  = 32          # skip degenerate crops smaller than this (px)

    # ── Loss weights ────────────────────────────────────────────────────────────
    FOCAL_ALPHA  = 2.0           # focal exponent — higher → harder focus
    LOSS_DICE_W  = 0.4           # weight of dice term; (1-LOSS_DICE_W) = focal-MSE

    # ── Inference (for post-training benchmark) ──────────────────────────────────
    DETECTION_THRESHOLD  = 0.10
    MIN_KEYPOINT_DISTANCE = 4

    @classmethod
    def setup(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.VIS_DIR,    exist_ok=True)
        return cls


# ================================================================================
# SE-UNet  (Squeeze-Excite attention blocks in the encoder)
# ================================================================================
class SEBlock(nn.Module):
    """Channel Squeeze-and-Excitation."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class DoubleConvSE(nn.Module):
    """Double conv + optional SE attention."""
    def __init__(self, in_ch: int, out_ch: int, use_se: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x))


class SEUNet(nn.Module):
    """
    U-Net with Squeeze-Excite attention in the encoder.
    Output shape identical to the original UNet → checkpoint drop-in compatible.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConvSE(in_channels, b,      use_se=False)  # skip SE on first block
        self.enc2 = DoubleConvSE(b,     b * 2,  use_se=True)
        self.enc3 = DoubleConvSE(b * 2, b * 4,  use_se=True)
        self.enc4 = DoubleConvSE(b * 4, b * 8,  use_se=True)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvSE(b * 8,  b * 16, use_se=True)

        self.up4  = nn.ConvTranspose2d(b * 16, b * 8,  2, stride=2)
        self.dec4 = DoubleConvSE(b * 16, b * 8,  use_se=False)
        self.up3  = nn.ConvTranspose2d(b * 8,  b * 4,  2, stride=2)
        self.dec3 = DoubleConvSE(b * 8,  b * 4,  use_se=False)
        self.up2  = nn.ConvTranspose2d(b * 4,  b * 2,  2, stride=2)
        self.dec2 = DoubleConvSE(b * 4,  b * 2,  use_se=False)
        self.up1  = nn.ConvTranspose2d(b * 2,  b,      2, stride=2)
        self.dec1 = DoubleConvSE(b * 2,  b,      use_se=False)

        self.out_channels = b

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1


# ================================================================================
# Loss functions
# ================================================================================
class FocalHeatmapLoss(nn.Module):
    """
    Combined Focal-MSE + Dice loss for heatmap regression.

    Focal-MSE: (1 - pred)^alpha * (pred - gt)^2  at positive locations
                pred^alpha * (pred - gt)^2         at negative locations
    Dice:      1 - 2*sum(pred*gt) / (sum(pred)+sum(gt)+eps)
    """
    def __init__(self, alpha: float = 2.0, dice_weight: float = 0.4):
        super().__init__()
        self.alpha       = alpha
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # ── Focal-MSE ────────────────────────────────────────────────────────
        pos_mask = (gt >= 0.5).float()
        neg_mask = 1.0 - pos_mask

        focal_pos = ((1.0 - pred) ** self.alpha) * ((pred - gt) ** 2) * pos_mask
        focal_neg = (pred ** self.alpha)          * ((pred - gt) ** 2) * neg_mask
        focal_mse = (focal_pos + focal_neg).mean()

        # ── Dice ─────────────────────────────────────────────────────────────
        eps    = 1e-6
        inter  = (pred * gt).sum(dim=(-1, -2))
        union  = pred.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2))
        dice   = 1.0 - (2.0 * inter + eps) / (union + eps)
        dice   = dice.mean()

        return (1.0 - self.dice_weight) * focal_mse + self.dice_weight * dice


# ================================================================================
# Augmentation pipelines
# ================================================================================
def _build_train_transform_albumentations(input_size: int):
    """Strong augmentation via albumentations — correct keypoint handling."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08, scale_limit=0.25, rotate_limit=30,
                border_mode=0, p=0.7
            ),
            A.Perspective(scale=(0.04, 0.10), p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.4),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.6),
            A.CoarseDropout(
                max_holes=6, max_height=30, max_width=30,
                min_holes=1, fill_value=0, p=0.3
            ),
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=False, angle_in_degrees=False
        ),
    )


def _build_val_transform_albumentations(input_size: int):
    return A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=False, angle_in_degrees=False
        ),
    )


class BasicTransform:
    """Fallback when albumentations is not available."""
    def __init__(self, input_size: int, augment: bool):
        ops = [T.Resize((input_size, input_size))]
        if augment:
            ops += [
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            ]
        ops += [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        self.tf = T.Compose(ops)

    def __call__(self, image_np, keypoints):
        pil = Image.fromarray(image_np)
        return self.tf(pil), keypoints


# ================================================================================
# Crop-Aware Dataset
# ================================================================================
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class CropAwareKeypointDataset(Dataset):
    """
    COCO-format keypoint dataset that crops each annotated region using its
    bbox (+ pad_ratio margin), then remaps keypoints into crop-space.

    This matches exactly what infer_two_stage_video.py does at runtime —
    eliminating the train/inference distribution gap.
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        indices: List[int],
        input_size: int = 384,
        num_keypoints: int = 4,
        sigma: float = 8.0,
        pad_ratio: float = 0.15,
        augment: bool = True,
    ):
        self.image_dir     = image_dir
        self.input_size    = input_size
        self.num_keypoints = num_keypoints
        self.sigma         = sigma
        self.pad_ratio     = pad_ratio
        self.augment       = augment

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # Build lookup tables
        id2img  = {img["id"]: img for img in coco["images"]}
        id2anns: Dict[int, List] = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            id2anns.setdefault(iid, []).append(ann)

        # Build flat sample list — one sample per annotation
        all_samples = []
        for ann in coco["annotations"]:
            img_info = id2img.get(ann["image_id"])
            if img_info is None:
                continue
            kps_flat = ann.get("keypoints", [])
            if len(kps_flat) < num_keypoints * 3:
                continue
            all_samples.append((img_info, ann))

        # Subset by provided indices, then filter by image existence on disk
        candidates = [all_samples[i] for i in indices if i < len(all_samples)]
        self.samples = []
        skipped = 0
        for img_info, ann in candidates:
            img_path = os.path.join(image_dir, img_info["file_name"])
            if os.path.exists(img_path):
                self.samples.append((img_info, ann))
            else:
                skipped += 1
        if skipped:
            print(f"  WARN: skipped {skipped} annotations (image not found on disk)")

        # Build augmentation transform
        if ALBUMENTATIONS_AVAILABLE:
            if augment:
                self._transform = _build_train_transform_albumentations(input_size)
            else:
                self._transform = _build_val_transform_albumentations(input_size)
            self._use_alb = True
        else:
            self._transform = BasicTransform(input_size, augment)
            self._use_alb   = False

        print(f"  Dataset: {len(self.samples)} samples | "
              f"aug={'ON' if augment else 'OFF'} | "
              f"alb={ALBUMENTATIONS_AVAILABLE}")

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _load_crop(self, img_info, ann):
        """Load image, crop around bbox+padding.  Returns (crop_rgb, meta)."""
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        ih, iw = img.shape[:2]

        bbox = ann.get("bbox")  # COCO: [x, y, w, h]
        if bbox and len(bbox) == 4:
            bx, by, bw, bh = bbox
            pad = int(max(bw, bh) * self.pad_ratio)
            x0 = int(_clamp(bx - pad, 0, iw - 1))
            y0 = int(_clamp(by - pad, 0, ih - 1))
            x1 = int(_clamp(bx + bw + pad, 1, iw))
            y1 = int(_clamp(by + bh + pad, 1, ih))
        else:
            x0, y0, x1, y1 = 0, 0, iw, ih

        crop = img[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        if ch < Config.MIN_CROP_SIZE or cw < Config.MIN_CROP_SIZE:
            # degenerate crop — fall back to full frame
            crop = img
            x0, y0, cw, ch = 0, 0, iw, ih

        return crop, {"x0": x0, "y0": y0, "cw": x1 - x0, "ch": y1 - y0}

    def _parse_keypoints(self, ann, meta):
        """
        Parse COCO keypoints, remap into crop-pixel space.
        Returns list of [cx, cy] in crop-pixel coordinates or [-1,-1] if invisible.
        Scaling to input_size is handled later by the augmentation pipeline.
        """
        kps_flat = ann["keypoints"]
        cw, ch   = meta["cw"], meta["ch"]
        x0, y0   = meta["x0"], meta["y0"]

        result = []
        for i in range(self.num_keypoints):
            base = i * 3
            if base + 2 >= len(kps_flat):
                result.append([-1.0, -1.0])
                continue
            kx, ky, vis = kps_flat[base], kps_flat[base + 1], kps_flat[base + 2]
            if vis <= 0:
                result.append([-1.0, -1.0])
                continue
            # to crop-pixel space (NOT scaled to input_size yet)
            cx = kx - x0
            cy = ky - y0
            # only keep if within crop bounds
            if 0 <= cx <= cw and 0 <= cy <= ch:
                result.append([float(cx), float(cy)])
            else:
                result.append([-1.0, -1.0])
        return result

    def _make_heatmap(self, keypoints, size: int) -> torch.Tensor:
        """Gaussian heatmap tensor (C, H, W)."""
        hm = torch.zeros(self.num_keypoints, size, size)
        xs = torch.arange(size, dtype=torch.float32)
        ys = torch.arange(size, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        for ki, kp in enumerate(keypoints):
            if kp[0] < 0:
                continue
            gx, gy = float(kp[0]), float(kp[1])
            g = torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * self.sigma ** 2))
            hm[ki] = torch.maximum(hm[ki], g)
        return hm

    # ── albumentations path ───────────────────────────────────────────────────
    def _apply_alb(self, crop_np, kps_scaled):
        """Apply albumentations transform with keypoint-aware augmentation."""
        # Build list of (x,y) for valid keypoints; track which are valid
        valid_xy = []
        valid_idx = []
        for i, kp in enumerate(kps_scaled):
            if kp[0] >= 0:
                valid_xy.append((float(kp[0]), float(kp[1])))
                valid_idx.append(i)

        # Resize crop to input_size before passing (alb does final resize)
        h, w = crop_np.shape[:2]

        result = self._transform(image=crop_np, keypoints=valid_xy)
        img_t  = result["image"]  # tensor CHW float
        out_kps_xy = result["keypoints"]

        # Rebuild full kp list
        final_kps = [[-1.0, -1.0]] * self.num_keypoints
        for j, idx in enumerate(valid_idx):
            if j < len(out_kps_xy):
                ox, oy = out_kps_xy[j]
                # clamp to [0, input_size-1]
                ox = float(_clamp(ox, 0.0, self.input_size - 1))
                oy = float(_clamp(oy, 0.0, self.input_size - 1))
                final_kps[idx] = [ox, oy]

        return img_t, final_kps

    # ── basic transform path ──────────────────────────────────────────────────
    def _apply_basic(self, crop_np, kps_crop):
        h, w = crop_np.shape[:2]
        img_t, _ = self._transform(crop_np, kps_crop)
        # Manually scale keypoints from crop-pixel space to input_size space
        scaled = []
        for kp in kps_crop:
            if kp[0] < 0:
                scaled.append([-1.0, -1.0])
            else:
                scaled.append([kp[0] / w * self.input_size,
                               kp[1] / h * self.input_size])
        return img_t, scaled

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info, ann = self.samples[idx]

        # Load & crop
        crop_np, meta = self._load_crop(img_info, ann)

        # Parse keypoints in crop-space
        kps_scaled = self._parse_keypoints(ann, meta)

        # Augment
        if self._use_alb:
            img_t, kps_aug = self._apply_alb(crop_np, kps_scaled)
        else:
            img_t, kps_aug = self._apply_basic(crop_np, kps_scaled)

        # Generate ground-truth heatmap
        hm = self._make_heatmap(kps_aug, self.input_size)

        return img_t, hm


# ================================================================================
# Lightning Module
# ================================================================================
class KeypointLightningModel(pl.LightningModule):
    def __init__(
        self,
        n_keypoints: int  = 4,
        base_channels: int = 64,
        lr: float          = 2e-4,
        weight_decay: float = 1e-4,
        max_epochs: int    = 100,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.max_epochs    = max_epochs
        self.warmup_epochs = warmup_epochs

        self.backbone = SEUNet(in_channels=3, base_channels=base_channels)
        self.head     = nn.Conv2d(self.backbone.out_channels, n_keypoints, kernel_size=1)
        nn.init.constant_(self.head.bias, -4.0)

        self.criterion = FocalHeatmapLoss(
            alpha=Config.FOCAL_ALPHA,
            dice_weight=Config.LOSS_DICE_W,
        )

        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []

    def forward(self, x):
        return torch.sigmoid(self.head(self.backbone(x)))

    # ── steps ────────────────────────────────────────────────────────────────
    def _shared_step(self, batch):
        imgs, hm_gt = batch
        hm_pred = self(imgs)
        return self.criterion(hm_pred, hm_gt)

    def training_step(self, batch, _):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        v = self.trainer.callback_metrics.get("train_loss_epoch")
        if v is not None:
            self.train_losses.append(float(v))

    def on_validation_epoch_end(self):
        v = self.trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_losses.append(float(v))

    # ── optimiser + scheduler ────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Linear warmup then cosine annealing
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.warmup_epochs))
            progress = (epoch - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ================================================================================
# Data split helper
# ================================================================================
def make_split_indices(n_total: int, train_ratio: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    n_train = int(n_total * train_ratio)
    return idx[:n_train], idx[n_train:]


# ================================================================================
# Keypoint extraction (for post-training eval)
# ================================================================================
from scipy.ndimage import maximum_filter

def extract_best_kp(hm_np: np.ndarray, threshold=0.10, min_dist=4):
    results = []
    for ch in range(hm_np.shape[0]):
        ch_data  = hm_np[ch]
        loc_max  = maximum_filter(ch_data, size=min_dist * 2 + 1)
        peaks    = (ch_data == loc_max) & (ch_data > threshold)
        coords   = np.where(peaks)
        if len(coords[0]) == 0:
            results.append(None)
            continue
        vals = ch_data[coords]
        idx  = np.argmax(vals)
        results.append((int(coords[1][idx]), int(coords[0][idx]), float(vals[idx])))
    return results


# ================================================================================
# Visualization helpers
# ================================================================================
KEYPOINT_NAMES  = ["top_left", "top_right", "bottom_right", "bottom_left"]
COLORS_HEX      = ["#FF3333", "#33FF66", "#3399FF", "#FFD700"]


def save_prediction_grid(imgs, hm_preds, hm_gts, save_path, n=4):
    """Save a grid of predictions vs ground-truth heatmaps."""
    n = min(n, len(imgs))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None]

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i in range(n):
        img_np = imgs[i].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)

        hm_p = hm_preds[i].detach().cpu().numpy()
        hm_g = hm_gts[i].cpu().numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input crop", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.max(hm_g, axis=0), cmap="hot", vmin=0, vmax=1)
        axes[i, 1].set_title("GT heatmap (max)", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(np.max(hm_p, axis=0), cmap="hot", vmin=0, vmax=1)
        axes[i, 2].set_title("Pred heatmap (max)", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_sample_overlays(dataset, save_dir, n_samples=5):
    """
    Save N individual sample visualizations with GT heatmap overlaid on the
    input crop.  Each image has 3 panels:
      [Input crop]  |  [GT heatmap]  |  [Overlay + keypoint markers]
    This makes alignment issues immediately obvious before training starts.
    """
    n_samples = min(n_samples, len(dataset))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    kp_colors = ["red", "lime", "deepskyblue", "gold"]

    for si in range(n_samples):
        img_t, hm_t = dataset[si]

        # De-normalise image
        img_np = img_t.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)

        hm_np  = hm_t.numpy()                       # (C, H, W)
        hm_max = np.max(hm_np, axis=0)               # (H, W)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Input crop
        axes[0].imshow(img_np)
        axes[0].set_title(f"Sample {si} — Input crop", fontsize=10)
        axes[0].axis("off")

        # Panel 2: GT heatmap alone
        axes[1].imshow(hm_max, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("GT heatmap (max)", fontsize=10)
        axes[1].axis("off")

        # Panel 3: Overlay — crop + heatmap blend + keypoint markers
        axes[2].imshow(img_np)
        axes[2].imshow(hm_max, cmap="hot", alpha=0.45, vmin=0, vmax=1)
        for ki in range(hm_np.shape[0]):
            ch = hm_np[ki]
            if ch.max() < 0.1:
                continue
            # Find peak location
            peak_yx = np.unravel_index(np.argmax(ch), ch.shape)
            py, px = peak_yx
            label = KEYPOINT_NAMES[ki] if ki < len(KEYPOINT_NAMES) else f"kp{ki}"
            axes[2].plot(px, py, 'o', color=kp_colors[ki % len(kp_colors)],
                         markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            axes[2].text(px + 6, py - 6, label, fontsize=7,
                         color=kp_colors[ki % len(kp_colors)],
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
        axes[2].set_title("Overlay: crop + GT heatmap + KP", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        path = os.path.join(save_dir, f"sanity_sample_{si}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_training_curves(train_losses, val_losses, save_path):
    if not train_losses:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train", linewidth=2)
    if val_losses:
        ax.plot(range(len(val_losses)), val_losses, label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves (Focal-MSE + Dice)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ================================================================================
# Eval: per-keypoint mean distance error on the val set
# ================================================================================
@torch.no_grad()
def evaluate_model(model, val_loader, device):
    model.eval()
    errors_per_kp = [[] for _ in range(Config.NUM_KEYPOINTS)]
    detected      = [0]  * Config.NUM_KEYPOINTS
    total         = [0]  * Config.NUM_KEYPOINTS

    for imgs, hm_gts in val_loader:
        imgs   = imgs.to(device)
        hm_p   = model(imgs).cpu().numpy()
        hm_g   = hm_gts.numpy()

        for b in range(len(imgs)):
            pred_kps = extract_best_kp(hm_p[b], Config.DETECTION_THRESHOLD,
                                       Config.MIN_KEYPOINT_DISTANCE)
            from scipy.ndimage import maximum_filter as mf
            gt_kps   = extract_best_kp(hm_g[b], 0.5, 2)

            for ki in range(Config.NUM_KEYPOINTS):
                if gt_kps[ki] is not None:
                    total[ki] += 1
                    if pred_kps[ki] is not None:
                        detected[ki] += 1
                        dx = pred_kps[ki][0] - gt_kps[ki][0]
                        dy = pred_kps[ki][1] - gt_kps[ki][1]
                        errors_per_kp[ki].append(math.sqrt(dx*dx + dy*dy))

    print("\n  Per-keypoint evaluation (on val set):")
    print(f"  {'Keypoint':<15} {'Recall':>8} {'Mean Err (px)':>15}")
    print("  " + "-" * 42)
    all_errors = []
    for ki in range(Config.NUM_KEYPOINTS):
        recall = detected[ki] / max(1, total[ki])
        mean_e = float(np.mean(errors_per_kp[ki])) if errors_per_kp[ki] else float("nan")
        all_errors.extend(errors_per_kp[ki])
        print(f"  {KEYPOINT_NAMES[ki]:<15} {recall:>7.1%}  {mean_e:>12.2f}")
    overall = float(np.mean(all_errors)) if all_errors else float("nan")
    print(f"  {'OVERALL':<15} {'':>8} {overall:>12.2f}")
    return overall


# ================================================================================
# Main
# ================================================================================
def main():
    print("=" * 70)
    print("   CROP-AWARE KEYPOINT TRAINING  (SE-UNet + Focal-Dice Loss)")
    print("=" * 70)
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    Config.setup()
    pl.seed_everything(Config.SEED)

    # ── Device ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print()

    # ── Validate data paths ──────────────────────────────────────────────────
    if not os.path.exists(Config.DATA_JSON):
        print(f"ERROR: Annotation file not found:\n  {Config.DATA_JSON}")
        sys.exit(1)
    if not os.path.exists(Config.IMAGE_DIR):
        print(f"ERROR: Image directory not found:\n  {Config.IMAGE_DIR}")
        sys.exit(1)

    # ── Count total samples ──────────────────────────────────────────────────
    with open(Config.DATA_JSON) as f:
        coco_tmp = json.load(f)
    n_total = len(coco_tmp["annotations"])
    print(f"  Total annotations: {n_total}")

    train_idx, val_idx = make_split_indices(n_total, Config.TRAIN_VAL_SPLIT, Config.SEED)
    print(f"  Train / Val split : {len(train_idx)} / {len(val_idx)}")
    print()

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("STEP 1: Building datasets")
    print("-" * 60)
    train_ds = CropAwareKeypointDataset(
        Config.DATA_JSON, Config.IMAGE_DIR, train_idx,
        input_size=Config.INPUT_SIZE, num_keypoints=Config.NUM_KEYPOINTS,
        sigma=Config.HEATMAP_SIGMA,   pad_ratio=Config.CROP_PAD_RATIO,
        augment=True,
    )
    val_ds = CropAwareKeypointDataset(
        Config.DATA_JSON, Config.IMAGE_DIR, val_idx,
        input_size=Config.INPUT_SIZE, num_keypoints=Config.NUM_KEYPOINTS,
        sigma=Config.HEATMAP_SIGMA,   pad_ratio=Config.CROP_PAD_RATIO,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=(device == "cuda"),
        persistent_workers=(Config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=(device == "cuda"),
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    # ── Sanity-check: per-sample overlay visualizations ────────────────────
    n_sanity = min(5, len(train_ds))   # 2-5 samples for quick visual check
    print(f"  Sanity check — saving {n_sanity} sample overlays...")
    print("  (Check these images in outputs_cropped/visualizations/ before")
    print("   letting the full training run — terminate early if misaligned.)")
    save_sample_overlays(train_ds, Config.VIS_DIR, n_samples=n_sanity)

    # Also log tensor shapes from one batch
    sample_imgs, sample_hms = next(iter(train_loader))
    print(f"  Image tensor : {tuple(sample_imgs.shape)} {sample_imgs.dtype}")
    print(f"  Heatmap tensor: {tuple(sample_hms.shape)} "
          f"max={sample_hms.max():.3f}")
    save_prediction_grid(
        sample_imgs, sample_hms, sample_hms,
        os.path.join(Config.VIS_DIR, "sanity_check_batch.png"),
    )
    print()

    # ── Model ────────────────────────────────────────────────────────────────
    print("STEP 2: Building model")
    print("-" * 60)
    lit_model = KeypointLightningModel(
        n_keypoints  = Config.NUM_KEYPOINTS,
        base_channels= Config.BASE_CHANNELS,
        lr           = Config.LEARNING_RATE,
        weight_decay = Config.WEIGHT_DECAY,
        max_epochs   = Config.MAX_EPOCHS,
        warmup_epochs= 5,
    )
    n_params = sum(p.numel() for p in lit_model.parameters()) / 1e6
    print(f"  SE-UNet params: {n_params:.1f} M")
    print(f"  Loss: FocalMSE (α={Config.FOCAL_ALPHA}) + Dice (w={Config.LOSS_DICE_W})")
    print(f"  LR  : {Config.LEARNING_RATE} with cosine annealing + 5-epoch warmup")
    print()

    # ── Callbacks ────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(Config.OUTPUT_DIR, "checkpoints")
    ckpt_cb  = ModelCheckpoint(
        dirpath   = ckpt_dir,
        filename  = "best-{epoch:03d}-{val_loss:.4f}",
        monitor   = "val_loss",
        save_top_k = 3,
        mode      = "min",
        save_last = True,
    )
    early_cb = EarlyStopping(
        monitor="val_loss", patience=20, mode="min", verbose=True
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(Config.OUTPUT_DIR, name="logs")

    trainer = pl.Trainer(
        max_epochs   = Config.MAX_EPOCHS,
        accelerator  = "gpu" if torch.cuda.is_available() else "cpu",
        devices      = 1,
        callbacks    = [ckpt_cb, early_cb, lr_cb],
        logger       = logger,
        log_every_n_steps = max(1, len(train_loader) // 4),
        enable_progress_bar = True,
        precision    = "16-mixed" if torch.cuda.is_available() else 32,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("STEP 3: Training")
    print("-" * 60)
    t0 = time.time()
    trainer.fit(lit_model, train_loader, val_loader)
    train_time = time.time() - t0
    print(f"\n  Training complete: {train_time:.0f}s ({train_time/60:.1f} min)")

    # ── Save final checkpoint ─────────────────────────────────────────────────
    final_ckpt = os.path.join(Config.OUTPUT_DIR, "final_model.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"  Final checkpoint: {final_ckpt}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nSTEP 4: Generating training curves")
    print("-" * 60)
    plot_training_curves(
        lit_model.train_losses, lit_model.val_losses,
        os.path.join(Config.VIS_DIR, "training_curves.png"),
    )

    # ── Val visualizations ────────────────────────────────────────────────────
    print("\nSTEP 5: Validation visualizations")
    print("-" * 60)
    lit_model.eval().to(device)
    val_imgs, val_hms = next(iter(val_loader))
    with torch.no_grad():
        val_preds = lit_model(val_imgs.to(device))
    save_prediction_grid(
        val_imgs, val_preds.cpu(), val_hms,
        os.path.join(Config.VIS_DIR, "val_predictions.png"),
        n=min(4, len(val_imgs)),
    )

    # ── Per-keypoint evaluation ───────────────────────────────────────────────
    print("\nSTEP 6: Evaluation")
    print("-" * 60)
    mean_err = evaluate_model(lit_model, val_loader, device)

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    results = {
        "timestamp"     : datetime.now().isoformat(),
        "model"         : "SE-UNet",
        "loss"          : "FocalMSE+Dice",
        "input_size"    : Config.INPUT_SIZE,
        "crop_pad_ratio": Config.CROP_PAD_RATIO,
        "epochs_run"    : len(lit_model.train_losses),
        "train_time_s"  : round(train_time, 1),
        "final_val_loss": float(lit_model.val_losses[-1]) if lit_model.val_losses else None,
        "mean_kp_error_px": round(mean_err, 2) if not math.isnan(mean_err) else None,
        "best_ckpt"     : ckpt_cb.best_model_path,
        "final_ckpt"    : final_ckpt,
    }
    metrics_path = os.path.join(Config.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  DONE!")
    print(f"  Final model  : {final_ckpt}")
    print(f"  Best ckpt    : {ckpt_cb.best_model_path}")
    print(f"  Visuals      : {Config.VIS_DIR}")
    print(f"  Metrics      : {metrics_path}")
    print(f"\n  ► Use with infer_two_stage_video.py:")
    print(f"    --kp-weights {final_ckpt}")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    main()
