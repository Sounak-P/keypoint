"""
RTMPose-Style Keypoint Detection Training and Inference Script
==============================================================
Architecture: CSPNeXt Backbone + SimCC Coordinate Classification Head
STANDALONE with visualizations and metric graphs — direct comparison with UNet baseline
"""

import os
import sys
import json
import time
import glob
import math
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================
class Config:
    """Configuration class"""
    
    # Paths
    BASE_DIR = "/kaggle/working"
    DATA_DIR = "/kaggle/input/datasets/sounakp/keypointpapercorner/data"
    OUTPUT_DIR = "/kaggle/working/outputs_rtmpose"
    VIS_DIR = "/kaggle/working/outputs_rtmpose/visualizations"
    
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "valid")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Model
    INPUT_SIZE = 256
    NUM_KEYPOINTS = 4
    SIMCC_SPLIT_RATIO = 2  # Sub-pixel resolution multiplier
    
    # CSPNeXt backbone channels
    BACKBONE_CHANNELS = [64, 128, 256, 512]
    
    # Training
    MAX_EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    NUM_WORKERS = 2
    SEED = 42
    
    # SimCC
    SIMCC_SIGMA = 6.0  # Sigma for 1D Gaussian label encoding
    
    # Inference
    DETECTION_THRESHOLD = 0.1
    
    @classmethod
    def setup(cls):
        if not os.path.exists("/kaggle"):
            print("Running locally")
            cls.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            cls.DATA_DIR = os.path.join(cls.BASE_DIR, "data")
            cls.OUTPUT_DIR = os.path.join(cls.BASE_DIR, "outputs_rtmpose")
            cls.VIS_DIR = os.path.join(cls.OUTPUT_DIR, "visualizations")
            cls.TRAIN_DIR = os.path.join(cls.DATA_DIR, "train")
            cls.VAL_DIR = os.path.join(cls.DATA_DIR, "valid")
            cls.TEST_DIR = os.path.join(cls.DATA_DIR, "test")
        else:
            print("Running on Kaggle")
        
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.VIS_DIR, exist_ok=True)
        return cls


# =============================================
# CSPNeXt Building Blocks
# =============================================
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class DarknetBottleneck(nn.Module):
    """Bottleneck block with depthwise separable convolution (CSPNeXt style)."""
    def __init__(self, channels, expansion=0.5, use_depthwise=True):
        super().__init__()
        mid = int(channels * expansion)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        if use_depthwise:
            self.conv2 = DepthwiseSeparableConv(mid, channels, kernel_size=5)
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True)
            )
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPLayer(nn.Module):
    """Cross-Stage Partial layer with N bottleneck blocks and channel attention."""
    def __init__(self, in_ch, out_ch, num_blocks=2, expansion=0.5, use_depthwise=True):
        super().__init__()
        mid = int(out_ch * expansion)
        
        # Main branch (goes through bottlenecks)
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            *[DarknetBottleneck(mid, expansion=1.0, use_depthwise=use_depthwise)
              for _ in range(num_blocks)]
        )
        
        # Short branch (skip)
        self.short_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        
        # Merge
        self.merge_conv = nn.Sequential(
            nn.Conv2d(mid * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        
        self.attn = ChannelAttention(out_ch)
    
    def forward(self, x):
        main = self.blocks(self.main_conv(x))
        short = self.short_conv(x)
        merged = torch.cat([main, short], dim=1)
        out = self.merge_conv(merged)
        return self.attn(out)


# =============================================
# CSPNeXt Backbone
# =============================================
class CSPNeXt(nn.Module):
    """
    CSPNeXt backbone inspired by RTMDet / RTMPose.
    4 stages with increasing channels and spatial downsampling.
    """
    def __init__(self, in_channels=3, channels=[64, 128, 256, 512], num_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        # Stem: aggressive downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0] // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0] // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0] // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0] // 2, channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        for i in range(len(channels)):
            stage_in = channels[i] if i == 0 else channels[i - 1]
            layers = []
            if i > 0:
                # Downsample with strided conv
                layers.append(nn.Conv2d(stage_in, channels[i], 3, stride=2, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(channels[i]))
                layers.append(nn.SiLU(inplace=True))
            layers.append(CSPLayer(channels[i], channels[i], num_blocks=num_blocks[i]))
            self.stages.append(nn.Sequential(*layers))
        
        self.out_channels = channels[-1]
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


# =============================================
# SimCC Head
# =============================================
class SimCCHead(nn.Module):
    """
    SimCC (Simple Coordinate Classification) head.
    Reformulates keypoint localization as two 1D classification problems (x, y).

    For each keypoint, predicts:
      - x_logits: (W * split_ratio,) bins for horizontal coordinate
      - y_logits: (H * split_ratio,) bins for vertical coordinate
    """
    def __init__(self, in_channels, num_keypoints, input_size, simcc_split_ratio=2, hidden_dim=256):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        
        self.x_bins = int(input_size * simcc_split_ratio)
        self.y_bins = int(input_size * simcc_split_ratio)
        
        # Feature compression: 1x1 conv + GAP
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Gated Attention Unit (simplified)
        self.gau = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Classification heads
        self.fc_x = nn.Linear(hidden_dim, num_keypoints * self.x_bins)
        self.fc_y = nn.Linear(hidden_dim, num_keypoints * self.y_bins)
    
    def forward(self, x):
        """
        Args:
            x: backbone features (B, C, H, W)
        Returns:
            x_logits: (B, num_keypoints, x_bins)
            y_logits: (B, num_keypoints, y_bins)
        """
        B = x.shape[0]
        feat = self.feature_conv(x)
        feat = self.gap(feat).view(B, -1)
        
        # Gated attention
        gate = self.gau(feat)
        feat = feat * gate
        
        x_logits = self.fc_x(feat).view(B, self.num_keypoints, self.x_bins)
        y_logits = self.fc_y(feat).view(B, self.num_keypoints, self.y_bins)
        
        return x_logits, y_logits


# =============================================
# SimCC Label Encoding / Decoding
# =============================================
def generate_simcc_label(keypoints, input_size, simcc_split_ratio, sigma, num_keypoints):
    """
    Generate 1D Gaussian labels for SimCC.
    
    Args:
        keypoints: list of [x, y] pairs (pixel coords in input_size space)
        input_size: image size
        simcc_split_ratio: sub-pixel multiplier
        sigma: Gaussian sigma for label smoothing
        num_keypoints: number of keypoints
    
    Returns:
        x_labels: (num_keypoints, x_bins) — normalized 1D Gaussian
        y_labels: (num_keypoints, y_bins) — normalized 1D Gaussian
        valid_mask: (num_keypoints,) — 1 if keypoint is valid, 0 otherwise
    """
    x_bins = int(input_size * simcc_split_ratio)
    y_bins = int(input_size * simcc_split_ratio)
    
    x_labels = np.zeros((num_keypoints, x_bins), dtype=np.float32)
    y_labels = np.zeros((num_keypoints, y_bins), dtype=np.float32)
    valid_mask = np.zeros(num_keypoints, dtype=np.float32)
    
    for kp_idx in range(num_keypoints):
        if kp_idx < len(keypoints):
            kp = keypoints[kp_idx]
            if kp[0] >= 0 and kp[1] >= 0 and kp[0] < input_size and kp[1] < input_size:
                # Scale to bin space
                cx = kp[0] * simcc_split_ratio
                cy = kp[1] * simcc_split_ratio
                
                # Generate 1D Gaussians
                x_range = np.arange(x_bins, dtype=np.float32)
                y_range = np.arange(y_bins, dtype=np.float32)
                
                x_labels[kp_idx] = np.exp(-((x_range - cx) ** 2) / (2 * sigma ** 2))
                y_labels[kp_idx] = np.exp(-((y_range - cy) ** 2) / (2 * sigma ** 2))
                
                # Normalize to sum to 1 (probability distribution)
                x_sum = x_labels[kp_idx].sum()
                y_sum = y_labels[kp_idx].sum()
                if x_sum > 0:
                    x_labels[kp_idx] /= x_sum
                if y_sum > 0:
                    y_labels[kp_idx] /= y_sum
                
                valid_mask[kp_idx] = 1.0
    
    return x_labels, y_labels, valid_mask


def decode_simcc(x_logits, y_logits, simcc_split_ratio):
    """
    Decode SimCC logits to keypoint coordinates and confidences.
    
    Args:
        x_logits: (num_keypoints, x_bins) — raw logits
        y_logits: (num_keypoints, y_bins) — raw logits
        simcc_split_ratio: sub-pixel multiplier
    
    Returns:
        keypoints: list of (x, y) in input_size pixel space
        confidences: list of confidence scores
    """
    if isinstance(x_logits, torch.Tensor):
        x_probs = F.softmax(x_logits, dim=-1)
        y_probs = F.softmax(y_logits, dim=-1)
        x_probs = x_probs.detach().cpu().numpy()
        y_probs = y_probs.detach().cpu().numpy()
    else:
        x_probs = np.exp(x_logits - x_logits.max(axis=-1, keepdims=True))
        x_probs /= x_probs.sum(axis=-1, keepdims=True)
        y_probs = np.exp(y_logits - y_logits.max(axis=-1, keepdims=True))
        y_probs /= y_probs.sum(axis=-1, keepdims=True)
    
    num_keypoints = x_logits.shape[0]
    keypoints = []
    confidences = []
    
    for kp_idx in range(num_keypoints):
        x_bin = np.argmax(x_probs[kp_idx])
        y_bin = np.argmax(y_probs[kp_idx])
        
        x_conf = x_probs[kp_idx][x_bin]
        y_conf = y_probs[kp_idx][y_bin]
        conf = float(np.sqrt(x_conf * y_conf))  # geometric mean
        
        x_coord = x_bin / simcc_split_ratio
        y_coord = y_bin / simcc_split_ratio
        
        keypoints.append((float(x_coord), float(y_coord)))
        confidences.append(conf)
    
    return keypoints, confidences


# =============================================
# RTMPose Detector (Lightning Module)
# =============================================
class RTMPoseDetector(pl.LightningModule):
    def __init__(self, num_keypoints=4, input_size=256, simcc_split_ratio=2,
                 simcc_sigma=6.0, learning_rate=3e-4,
                 backbone_channels=[64, 128, 256, 512]):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.simcc_sigma = simcc_sigma
        self.learning_rate = learning_rate
        
        self.backbone = CSPNeXt(
            in_channels=3,
            channels=backbone_channels,
            num_blocks=[2, 2, 2, 2]
        )
        
        self.head = SimCCHead(
            in_channels=self.backbone.out_channels,
            num_keypoints=num_keypoints,
            input_size=input_size,
            simcc_split_ratio=simcc_split_ratio,
            hidden_dim=256
        )
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, x):
        features = self.backbone(x)
        x_logits, y_logits = self.head(features)
        return x_logits, y_logits
    
    def _compute_loss(self, x_logits, y_logits, keypoints_batch):
        """Compute KL-divergence loss between predicted and GT distributions."""
        batch_size = x_logits.shape[0]
        total_loss = 0.0
        valid_count = 0
        
        for i in range(batch_size):
            x_labels, y_labels, valid_mask = generate_simcc_label(
                keypoints_batch[i], self.input_size,
                self.simcc_split_ratio, self.simcc_sigma, self.num_keypoints
            )
            
            x_labels_t = torch.tensor(x_labels, device=self.device)
            y_labels_t = torch.tensor(y_labels, device=self.device)
            valid_mask_t = torch.tensor(valid_mask, device=self.device)
            
            # Log-softmax of predictions
            x_log_probs = F.log_softmax(x_logits[i], dim=-1)
            y_log_probs = F.log_softmax(y_logits[i], dim=-1)
            
            # KL divergence: sum over bins, mean over valid keypoints
            for kp_idx in range(self.num_keypoints):
                if valid_mask_t[kp_idx] > 0:
                    # KL(target || pred) = sum(target * (log(target) - log(pred)))
                    # Using F.kl_div which expects log(pred) and target
                    kl_x = F.kl_div(x_log_probs[kp_idx], x_labels_t[kp_idx],
                                    reduction='sum', log_target=False)
                    kl_y = F.kl_div(y_log_probs[kp_idx], y_labels_t[kp_idx],
                                    reduction='sum', log_target=False)
                    total_loss += (kl_x + kl_y)
                    valid_count += 1
        
        if valid_count > 0:
            total_loss = total_loss / valid_count
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        images, keypoints_batch = batch
        x_logits, y_logits = self(images)
        loss = self._compute_loss(x_logits, y_logits, keypoints_batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, keypoints_batch = batch
        x_logits, y_logits = self(images)
        loss = self._compute_loss(x_logits, y_logits, keypoints_batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        if train_loss is not None:
            self.train_losses.append(float(train_loss))
    
    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


# =============================================
# Dataset
# =============================================
class KeypointDataset(Dataset):
    def __init__(self, data_dir: str, input_size: int = 256, num_keypoints: int = 4):
        self.data_dir = data_dir
        self.input_size = input_size
        self.num_keypoints = num_keypoints
        
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        self.image_files = sorted(self.image_files)
        
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        image = self.transform(image)
        
        img_filename = os.path.basename(img_path)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_filename)[0] + ".txt")
        
        keypoints = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                values = line.strip().split()
                if len(values) >= 5 + self.num_keypoints * 3:
                    for i in range(self.num_keypoints):
                        idx_start = 5 + i * 3
                        kp_x = float(values[idx_start]) * self.input_size
                        kp_y = float(values[idx_start + 1]) * self.input_size
                        keypoints.append([kp_x, kp_y])
                    break
        
        while len(keypoints) < self.num_keypoints:
            keypoints.append([-1, -1])
        
        return image, keypoints
    
    def get_original_image(self, idx):
        return Image.open(self.image_files[idx]).convert('RGB')
    
    def get_image_path(self, idx):
        return self.image_files[idx]


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    keypoints = [item[1] for item in batch]
    return images, keypoints


# =============================================
# Visualization Functions
# =============================================
def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        ax1.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (KL-Divergence)', fontsize=12)
    ax1.set_title('RTMPose Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    if train_losses and val_losses and len(train_losses) == len(val_losses):
        ratio = [v/t if t > 0 else 1 for v, t in zip(val_losses, train_losses)]
        ax2.plot(epochs, ratio, 'g-', linewidth=2)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Val/Train Loss Ratio', fontsize=12)
        ax2.set_title('Overfitting Monitor', fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


def plot_metrics_summary(metrics, save_path):
    """Plot metrics summary"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Timing
    ax1 = axes[0, 0]
    timing = metrics['timing']
    bars = ax1.bar(['Mean', 'Median', 'P95', 'P99'], 
                   [timing['mean_ms'], timing['median_ms'], timing['p95_ms'], timing['p99_ms']],
                   color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('RTMPose Inference Timing', fontsize=14)
    for bar, val in zip(bars, [timing['mean_ms'], timing['median_ms'], timing['p95_ms'], timing['p99_ms']]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', fontsize=10)
    
    # Detection metrics
    ax2 = axes[0, 1]
    det = metrics['detection']
    bars = ax2.bar(['Precision', 'Recall', 'F1'], 
                   [det['precision'], det['recall'], det['f1_score']],
                   color=['#9b59b6', '#1abc9c', '#34495e'])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Detection Metrics', fontsize=14)
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars, [det['precision'], det['recall'], det['f1_score']]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=10)
    
    # TP/FP/FN
    ax3 = axes[1, 0]
    labels = ['True Positives', 'False Positives', 'False Negatives']
    values = [det['true_positives'], det['false_positives'], det['false_negatives']]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    bars = ax3.bar(labels, values, color=colors)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Detection Breakdown', fontsize=14)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(val), ha='center', fontsize=10)
    
    # Throughput
    ax4 = axes[1, 1]
    th = metrics['throughput']
    ax4.pie([th['fps']], labels=[f"FPS: {th['fps']:.1f}"], 
           colors=['#3498db'], autopct='', startangle=90,
           wedgeprops={'width': 0.4})
    ax4.text(0, 0, f"{th['fps']:.1f}\nFPS", ha='center', va='center', fontsize=20, fontweight='bold')
    ax4.set_title('Throughput', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics summary saved: {save_path}")


def visualize_prediction(image, gt_keypoints, pred_keypoints, confidences, save_path):
    """Visualize predictions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_np = np.array(image)
    
    # Ground Truth
    ax1 = axes[0]
    ax1.imshow(img_np)
    colors_gt = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    for i, kp in enumerate(gt_keypoints):
        if kp[0] >= 0 and kp[1] >= 0:
            ax1.scatter(kp[0], kp[1], c=colors_gt[i % 4], s=100, marker='o', 
                       edgecolors='white', linewidths=2)
            ax1.annotate(f'GT{i}', (kp[0], kp[1]), xytext=(5, 5), 
                        textcoords='offset points', color='white', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor=colors_gt[i % 4], alpha=0.7))
    ax1.set_title('Ground Truth', fontsize=12)
    ax1.axis('off')
    
    # Predictions
    ax2 = axes[1]
    ax2.imshow(img_np)
    colors_pred = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for i, (kp, conf) in enumerate(zip(pred_keypoints, confidences)):
        ax2.scatter(kp[0], kp[1], c=colors_pred[i % 4], s=80, marker='x', linewidths=2)
        ax2.annotate(f'{conf:.2f}', (kp[0], kp[1]), xytext=(5, -10), 
                    textcoords='offset points', color='white', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor=colors_pred[i % 4], alpha=0.7))
    ax2.set_title('RTMPose Predictions', fontsize=12)
    ax2.axis('off')
    
    # Overlay
    ax3 = axes[2]
    ax3.imshow(img_np)
    for i, kp in enumerate(gt_keypoints):
        if kp[0] >= 0 and kp[1] >= 0:
            ax3.scatter(kp[0], kp[1], c='lime', s=120, marker='o', 
                       edgecolors='white', linewidths=2, label='GT' if i == 0 else '')
    for j, (kp, conf) in enumerate(zip(pred_keypoints, confidences)):
        ax3.scatter(kp[0], kp[1], c='red', s=80, marker='x', 
                   linewidths=2, label='Pred' if j == 0 else '')
    ax3.legend(loc='upper right')
    ax3.set_title('Overlay', fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_simcc_visualization(x_logits, y_logits, save_path, num_keypoints=4):
    """Visualize SimCC probability distributions for each keypoint."""
    fig, axes = plt.subplots(2, num_keypoints, figsize=(4 * num_keypoints, 6))
    
    if isinstance(x_logits, torch.Tensor):
        x_probs = F.softmax(x_logits, dim=-1).cpu().numpy()
        y_probs = F.softmax(y_logits, dim=-1).cpu().numpy()
    else:
        x_probs = x_logits
        y_probs = y_logits
    
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44']
    
    for kp_idx in range(num_keypoints):
        # X distribution
        ax_x = axes[0, kp_idx] if num_keypoints > 1 else axes[0]
        ax_x.plot(x_probs[kp_idx], color=colors[kp_idx % 4], linewidth=1.5)
        peak_x = np.argmax(x_probs[kp_idx])
        ax_x.axvline(x=peak_x, color='gray', linestyle='--', alpha=0.5)
        ax_x.set_title(f'KP{kp_idx} X-dist (peak={peak_x})', fontsize=9)
        ax_x.set_xlabel('X bin')
        ax_x.set_ylabel('Probability')
        
        # Y distribution
        ax_y = axes[1, kp_idx] if num_keypoints > 1 else axes[1]
        ax_y.plot(y_probs[kp_idx], color=colors[kp_idx % 4], linewidth=1.5)
        peak_y = np.argmax(y_probs[kp_idx])
        ax_y.axvline(x=peak_y, color='gray', linestyle='--', alpha=0.5)
        ax_y.set_title(f'KP{kp_idx} Y-dist (peak={peak_y})', fontsize=9)
        ax_y.set_xlabel('Y bin')
        ax_y.set_ylabel('Probability')
    
    plt.suptitle('SimCC Coordinate Distributions', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================
# Inference Engine
# =============================================
class InferenceEngine:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        orig_size = image.size
        image_resized = image.resize((Config.INPUT_SIZE, Config.INPUT_SIZE), Image.BILINEAR)
        input_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            x_logits, y_logits = self.model(input_tensor)
        
        keypoints, confidences = decode_simcc(
            x_logits[0], y_logits[0], Config.SIMCC_SPLIT_RATIO
        )
        
        return keypoints, confidences, x_logits[0], y_logits[0]
    
    def benchmark(self, dataset, num_samples=None, warmup=10, save_visualizations=True):
        if num_samples is None:
            num_samples = len(dataset)
        num_samples = min(num_samples, len(dataset))
        
        print(f"\n{'='*60}")
        print("RTMPOSE INFERENCE BENCHMARK")
        print("="*60)
        
        # Warmup
        print(f"Warming up ({warmup} runs)...")
        for i in range(min(warmup, len(dataset))):
            img, _ = dataset[i]
            with torch.no_grad():
                _ = self.model(img.unsqueeze(0).to(self.device))
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_times = []
        all_predictions = []
        all_ground_truths = []
        all_confidences = []
        memory_usage = []
        
        print(f"Benchmarking on {num_samples} samples...")
        for i in range(num_samples):
            img, gt_kps = dataset[i]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                x_logits, y_logits = self.model(img.unsqueeze(0).to(self.device))
            keypoints, confidences = decode_simcc(
                x_logits[0], y_logits[0], Config.SIMCC_SPLIT_RATIO
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            inference_times.append((end - start) * 1000)
            
            all_predictions.append(keypoints)
            all_ground_truths.append(gt_kps)
            all_confidences.append(confidences)
            
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1e6)
            
            # Save visualizations
            if save_visualizations and i < 10:
                orig_img = dataset.get_original_image(i)
                orig_w, orig_h = orig_img.size
                
                scale_x = orig_w / Config.INPUT_SIZE
                scale_y = orig_h / Config.INPUT_SIZE
                
                scaled_pred = [(int(kp[0] * scale_x), int(kp[1] * scale_y)) for kp in keypoints]
                
                scaled_gt = []
                for kp in gt_kps:
                    if kp[0] >= 0:
                        scaled_gt.append([kp[0] * scale_x, kp[1] * scale_y])
                    else:
                        scaled_gt.append(kp)
                
                scaled_confs = confidences
                
                vis_path = os.path.join(Config.VIS_DIR, f"pred_{i:03d}.png")
                visualize_prediction(orig_img, scaled_gt, scaled_pred, scaled_confs, vis_path)
                
                # SimCC distribution visualization
                simcc_path = os.path.join(Config.VIS_DIR, f"simcc_{i:03d}.png")
                create_simcc_visualization(x_logits[0], y_logits[0], simcc_path, Config.NUM_KEYPOINTS)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_samples}")
        
        # Compute metrics
        inference_times = np.array(inference_times)
        
        total_tp, total_fp, total_fn = 0, 0, 0
        distances = []
        
        for preds, gts in zip(all_predictions, all_ground_truths):
            valid_gts = [g for g in gts if g[0] >= 0 and g[1] >= 0]
            used = set()
            
            for pred in preds:
                min_dist = float('inf')
                min_idx = -1
                for j, gt in enumerate(valid_gts):
                    if j in used:
                        continue
                    dist = np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j
                
                if min_dist <= 8:
                    total_tp += 1
                    distances.append(min_dist)
                    used.add(min_idx)
                else:
                    total_fp += 1
            
            total_fn += len(valid_gts) - len(used)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "model": "RTMPose (CSPNeXt + SimCC)",
            "timing": {
                "mean_ms": float(np.mean(inference_times)),
                "std_ms": float(np.std(inference_times)),
                "min_ms": float(np.min(inference_times)),
                "max_ms": float(np.max(inference_times)),
                "median_ms": float(np.median(inference_times)),
                "p95_ms": float(np.percentile(inference_times, 95)),
                "p99_ms": float(np.percentile(inference_times, 99)),
            },
            "throughput": {
                "fps": float(1000 / np.mean(inference_times)),
                "total_images": num_samples,
                "total_time_sec": float(np.sum(inference_times) / 1000),
            },
            "memory": {
                "mean_mb": float(np.mean(memory_usage)) if memory_usage else 0,
                "max_mb": float(np.max(memory_usage)) if memory_usage else 0,
            },
            "detection": {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "mean_error_px": float(np.mean(distances)) if distances else 0,
            }
        }
        
        return metrics


def print_metrics(metrics, training_time=None):
    print("\n" + "="*60)
    print("RTMPOSE COMPREHENSIVE METRICS REPORT")
    print("="*60)
    print(f"Model: {metrics.get('model', 'RTMPose')}")
    
    if training_time:
        print(f"\n📊 TRAINING: {training_time:.2f}s ({training_time/60:.2f} min)")
    
    t = metrics["timing"]
    print(f"\n⏱️  INFERENCE TIMING")
    print(f"   Mean: {t['mean_ms']:.2f}ms | Std: {t['std_ms']:.2f}ms")
    print(f"   Min: {t['min_ms']:.2f}ms | Max: {t['max_ms']:.2f}ms | P95: {t['p95_ms']:.2f}ms")
    
    th = metrics["throughput"]
    print(f"\n🚀 THROUGHPUT: {th['fps']:.2f} FPS")
    
    m = metrics["memory"]
    print(f"\n💾 MEMORY: Mean {m['mean_mb']:.2f}MB | Max {m['max_mb']:.2f}MB")
    
    d = metrics["detection"]
    print(f"\n🎯 DETECTION")
    print(f"   Precision: {d['precision']:.4f} | Recall: {d['recall']:.4f} | F1: {d['f1_score']:.4f}")
    print(f"   TP: {d['true_positives']} | FP: {d['false_positives']} | FN: {d['false_negatives']}")
    print(f"   Mean Error: {d['mean_error_px']:.2f}px")
    print("="*60)


# =============================================
# Model Summary
# =============================================
def print_model_summary(model):
    """Print model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print("RTMPOSE MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Architecture: CSPNeXt Backbone + SimCC Head")
    print(f"Backbone channels: {Config.BACKBONE_CHANNELS}")
    print(f"SimCC split ratio: {Config.SIMCC_SPLIT_RATIO}x")
    print(f"SimCC bins: {Config.INPUT_SIZE * Config.SIMCC_SPLIT_RATIO} per axis")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    print(f"{'='*60}")


# =============================================
# Main
# =============================================
def main():
    print("="*60)
    print("RTMPOSE KEYPOINT DETECTION - TRAINING & INFERENCE")
    print("Architecture: CSPNeXt + SimCC (Coordinate Classification)")
    print("="*60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    Config.setup()
    pl.seed_everything(Config.SEED)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Datasets
    print("\n" + "-"*60)
    print("STEP 1: Loading datasets")
    print("-"*60)
    
    train_dataset = KeypointDataset(Config.TRAIN_DIR, Config.INPUT_SIZE, Config.NUM_KEYPOINTS)
    val_dataset = KeypointDataset(Config.VAL_DIR, Config.INPUT_SIZE, Config.NUM_KEYPOINTS)
    test_dataset = KeypointDataset(Config.TEST_DIR, Config.INPUT_SIZE, Config.NUM_KEYPOINTS)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, collate_fn=collate_fn)
    
    # Model
    print("\n" + "-"*60)
    print("STEP 2: Training RTMPose model")
    print("-"*60)
    
    model = RTMPoseDetector(
        num_keypoints=Config.NUM_KEYPOINTS,
        input_size=Config.INPUT_SIZE,
        simcc_split_ratio=Config.SIMCC_SPLIT_RATIO,
        simcc_sigma=Config.SIMCC_SIGMA,
        learning_rate=Config.LEARNING_RATE,
        backbone_channels=Config.BACKBONE_CHANNELS
    )
    
    print_model_summary(model)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(Config.OUTPUT_DIR, "checkpoints"),
        filename="rtmpose-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=False,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    logger = TensorBoardLogger(Config.OUTPUT_DIR, name="logs")
    
    # Lightweight epoch-end callback (avoids IOPub flooding on Kaggle)
    class EpochPrintCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            t_loss = trainer.callback_metrics.get('train_loss_epoch', float('nan'))
            v_loss = trainer.callback_metrics.get('val_loss', float('nan'))
            lr = trainer.optimizers[0].param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | train_loss: {t_loss:.4f} | val_loss: {v_loss:.4f} | lr: {lr:.2e}")
    
    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, EpochPrintCallback()],
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=False,
    )
    
    print("\nStarting training...")
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s ({training_time/60:.2f} min)")
    
    model_path = os.path.join(Config.OUTPUT_DIR, "final_rtmpose.ckpt")
    trainer.save_checkpoint(model_path)
    print(f"Model saved: {model_path}")
    
    # Training curves
    print("\n" + "-"*60)
    print("STEP 3: Generating training graphs")
    print("-"*60)
    
    if model.train_losses and model.val_losses:
        plot_training_curves(
            model.train_losses, 
            model.val_losses,
            os.path.join(Config.VIS_DIR, "training_curves.png")
        )
    
    # Inference
    print("\n" + "-"*60)
    print("STEP 4: Inference benchmark with visualizations")
    print("-"*60)
    
    engine = InferenceEngine(model, device)
    metrics = engine.benchmark(test_dataset, save_visualizations=True)
    
    plot_metrics_summary(metrics, os.path.join(Config.VIS_DIR, "metrics_summary.png"))
    
    print_metrics(metrics, training_time)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "RTMPose (CSPNeXt + SimCC)",
        "config": {
            "input_size": Config.INPUT_SIZE,
            "batch_size": Config.BATCH_SIZE,
            "max_epochs": Config.MAX_EPOCHS,
            "learning_rate": Config.LEARNING_RATE,
            "simcc_split_ratio": Config.SIMCC_SPLIT_RATIO,
            "simcc_sigma": Config.SIMCC_SIGMA,
            "backbone_channels": Config.BACKBONE_CHANNELS,
        },
        "training_time_sec": training_time,
        "metrics": metrics,
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Visualizations saved to: {Config.VIS_DIR}")
    print("\n" + "="*60)
    print("RTMPOSE COMPLETED SUCCESSFULLY!")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    metrics = main()
