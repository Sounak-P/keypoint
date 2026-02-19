"""
Kaggle Keypoint Detection Training and Inference Script
========================================================
STANDALONE with visualizations and metric graphs
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

# =============================================
# Configuration
# =============================================
class Config:
    """Configuration class"""
    
    # Paths - Updated for COCO format dataset
    BASE_DIR = "/kaggle/working"
    DATA_DIR = "/kaggle/input/keypointpapercorner/data"
    OUTPUT_DIR = "/kaggle/working/outputs"
    VIS_DIR = "/kaggle/working/outputs/visualizations"
    
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "valid")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Model
    INPUT_SIZE = 256
    NUM_KEYPOINTS = 4
    
    # Training
    MAX_EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    NUM_WORKERS = 2
    SEED = 42
    
    # Heatmap
    HEATMAP_SIGMA = 3
    
    # Inference
    MAX_KEYPOINTS = 20
    DETECTION_THRESHOLD = 0.1
    MIN_KEYPOINT_DISTANCE = 4
    
    @classmethod
    def setup(cls):
        if not os.path.exists("/kaggle"):
            print("Running locally")
            cls.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            cls.DATA_DIR = os.path.join(cls.BASE_DIR, "data")
            cls.OUTPUT_DIR = os.path.join(cls.BASE_DIR, "outputs")
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
# U-Net Backbone
# =============================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
        self.out_channels = base_channels
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return d1


# =============================================
# Keypoint Detector Model
# =============================================
class KeypointDetector(pl.LightningModule):
    def __init__(self, n_keypoints=4, heatmap_sigma=3, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_keypoints = n_keypoints
        self.heatmap_sigma = heatmap_sigma
        self.learning_rate = learning_rate
        
        self.backbone = UNet(in_channels=3, base_channels=64)
        self.head = nn.Conv2d(self.backbone.out_channels, n_keypoints, kernel_size=1)
        nn.init.constant_(self.head.bias, -4)
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        features = self.backbone(x)
        heatmaps = torch.sigmoid(self.head(features))
        return heatmaps
    
    def generate_heatmap(self, keypoints, height, width):
        heatmaps = torch.zeros(self.n_keypoints, height, width, device=self.device)
        
        for kp_idx in range(self.n_keypoints):
            if kp_idx < len(keypoints):
                kp = keypoints[kp_idx]
                if len(kp) >= 2 and kp[0] >= 0 and kp[1] >= 0:
                    x, y = kp[0], kp[1]
                    if 0 <= x < width and 0 <= y < height:
                        xx, yy = torch.meshgrid(
                            torch.arange(width, device=self.device, dtype=torch.float32),
                            torch.arange(height, device=self.device, dtype=torch.float32),
                            indexing='xy'
                        )
                        gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.heatmap_sigma**2))
                        heatmaps[kp_idx] = torch.maximum(heatmaps[kp_idx], gaussian)
        
        return heatmaps
    
    def training_step(self, batch, batch_idx):
        images, keypoints_batch = batch
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        
        pred_heatmaps = self(images)
        
        gt_heatmaps = torch.zeros_like(pred_heatmaps)
        for i in range(batch_size):
            gt_heatmaps[i] = self.generate_heatmap(keypoints_batch[i], height, width)
        
        loss = F.binary_cross_entropy(pred_heatmaps, gt_heatmaps)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, keypoints_batch = batch
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        
        pred_heatmaps = self(images)
        
        gt_heatmaps = torch.zeros_like(pred_heatmaps)
        for i in range(batch_size):
            gt_heatmaps[i] = self.generate_heatmap(keypoints_batch[i], height, width)
        
        loss = F.binary_cross_entropy(pred_heatmaps, gt_heatmaps)
        
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
# Dataset - COCO Format
# =============================================
class KeypointDataset(Dataset):
    def __init__(self, data_dir: str, input_size: int = 256, num_keypoints: int = 4):
        self.data_dir = data_dir
        self.input_size = input_size
        self.num_keypoints = num_keypoints
        
        # Load COCO annotations
        annotation_file = os.path.join(data_dir, "_annotations.coco.json")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image id to image info mapping
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        # Build image id to annotations mapping
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # List of image ids that we'll iterate over
        self.image_ids = list(self.images.keys())
        
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        
        print(f"Found {len(self.image_ids)} images in {data_dir} (COCO format)")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image - images are directly in the data_dir
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Apply transforms
        image = self.transform(image)
        
        # Get keypoints from annotations
        keypoints = []
        if img_id in self.annotations:
            # Take the first annotation for this image (assuming one object per image)
            ann = self.annotations[img_id][0]
            if 'keypoints' in ann:
                kps = ann['keypoints']
                # COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
                # where v is visibility (0=not labeled, 1=labeled but not visible, 2=visible)
                for i in range(0, len(kps), 3):
                    if i + 2 < len(kps):
                        x = kps[i]
                        y = kps[i + 1]
                        v = kps[i + 2]
                        
                        # Scale keypoints to input size
                        x_scaled = (x / orig_w) * self.input_size
                        y_scaled = (y / orig_h) * self.input_size
                        
                        if v > 0:  # If keypoint is labeled
                            keypoints.append([x_scaled, y_scaled])
                        else:
                            keypoints.append([-1, -1])
        
        # Pad with invalid keypoints if we don't have enough
        while len(keypoints) < self.num_keypoints:
            keypoints.append([-1, -1])
        
        return image, keypoints[:self.num_keypoints]
    
    def get_original_image(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        return Image.open(img_path).convert('RGB')
    
    def get_image_path(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        return os.path.join(self.data_dir, img_info['file_name'])


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    keypoints = [item[1] for item in batch]
    return images, keypoints


# =============================================
# Keypoint Extraction
# =============================================
def extract_keypoints_from_heatmap(heatmap, max_keypoints=20, threshold=0.1, min_distance=4):
    from scipy.ndimage import maximum_filter
    
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    keypoints = []
    confidences = []
    
    for channel_idx in range(heatmap.shape[0]):
        channel = heatmap[channel_idx]
        channel_kps = []
        channel_confs = []
        
        local_max = maximum_filter(channel, size=min_distance * 2 + 1)
        peaks = (channel == local_max) & (channel > threshold)
        
        peak_coords = np.where(peaks)
        peak_values = channel[peak_coords]
        
        sorted_indices = np.argsort(peak_values)[::-1][:max_keypoints]
        
        for idx in sorted_indices:
            y, x = peak_coords[0][idx], peak_coords[1][idx]
            conf = peak_values[idx]
            channel_kps.append((int(x), int(y)))
            channel_confs.append(float(conf))
        
        keypoints.append(channel_kps)
        confidences.append(channel_confs)
    
    return keypoints, confidences


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
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
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
    ax1.set_title('Inference Timing', fontsize=14)
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
    for ch_idx, (ch_kps, ch_confs) in enumerate(zip(pred_keypoints, confidences)):
        for i, (kp, conf) in enumerate(zip(ch_kps, ch_confs)):
            ax2.scatter(kp[0], kp[1], c=colors_pred[ch_idx % 4], s=80, marker='x', 
                       linewidths=2)
            ax2.annotate(f'{conf:.2f}', (kp[0], kp[1]), xytext=(5, -10), 
                        textcoords='offset points', color='white', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor=colors_pred[ch_idx % 4], alpha=0.7))
    ax2.set_title('Predictions', fontsize=12)
    ax2.axis('off')
    
    # Overlay
    ax3 = axes[2]
    ax3.imshow(img_np)
    for i, kp in enumerate(gt_keypoints):
        if kp[0] >= 0 and kp[1] >= 0:
            ax3.scatter(kp[0], kp[1], c='lime', s=120, marker='o', 
                       edgecolors='white', linewidths=2, label='GT' if i == 0 else '')
    for ch_kps, ch_confs in zip(pred_keypoints, confidences):
        for j, (kp, conf) in enumerate(zip(ch_kps, ch_confs)):
            ax3.scatter(kp[0], kp[1], c='red', s=80, marker='x', 
                       linewidths=2, label='Pred' if j == 0 else '')
    ax3.legend(loc='upper right')
    ax3.set_title('Overlay', fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_heatmap_visualization(heatmaps, save_path):
    """Visualize heatmaps"""
    n_channels = heatmaps.shape[0]
    fig, axes = plt.subplots(1, n_channels + 1, figsize=(4 * (n_channels + 1), 4))
    
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    
    for i in range(n_channels):
        axes[i].imshow(heatmaps[i], cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(f'Keypoint {i}', fontsize=10)
        axes[i].axis('off')
    
    combined = np.max(heatmaps, axis=0)
    axes[-1].imshow(combined, cmap='hot', vmin=0, vmax=1)
    axes[-1].set_title('Combined', fontsize=10)
    axes[-1].axis('off')
    
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
            heatmaps = self.model(input_tensor)
        
        keypoints, confidences = extract_keypoints_from_heatmap(
            heatmaps[0], Config.MAX_KEYPOINTS, Config.DETECTION_THRESHOLD, Config.MIN_KEYPOINT_DISTANCE
        )
        
        return keypoints, confidences, heatmaps[0]
    
    def benchmark(self, dataset, num_samples=None, warmup=10, save_visualizations=True):
        if num_samples is None:
            num_samples = len(dataset)
        num_samples = min(num_samples, len(dataset))
        
        print(f"\n{'='*60}")
        print("INFERENCE BENCHMARK")
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
                heatmaps = self.model(img.unsqueeze(0).to(self.device))
            keypoints, confidences = extract_keypoints_from_heatmap(heatmaps[0])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            inference_times.append((end - start) * 1000)
            
            pred_flat = []
            for ch_kps in keypoints:
                pred_flat.extend(ch_kps)
            
            all_predictions.append(pred_flat)
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
                
                scaled_pred = []
                for ch_kps in keypoints:
                    scaled_ch = [(int(kp[0] * scale_x), int(kp[1] * scale_y)) for kp in ch_kps]
                    scaled_pred.append(scaled_ch)
                
                scaled_gt = []
                for kp in gt_kps:
                    if kp[0] >= 0:
                        scaled_gt.append([kp[0] * scale_x, kp[1] * scale_y])
                    else:
                        scaled_gt.append(kp)
                
                vis_path = os.path.join(Config.VIS_DIR, f"pred_{i:03d}.png")
                visualize_prediction(orig_img, scaled_gt, scaled_pred, confidences, vis_path)
                
                heatmap_path = os.path.join(Config.VIS_DIR, f"heatmap_{i:03d}.png")
                create_heatmap_visualization(heatmaps[0], heatmap_path)
            
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
    print("COMPREHENSIVE METRICS REPORT")
    print("="*60)
    
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
# Main
# =============================================
def main():
    print("="*60)
    print("KEYPOINT DETECTION - TRAINING & INFERENCE")
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
    print("STEP 2: Training model")
    print("-"*60)
    
    model = KeypointDetector(
        n_keypoints=Config.NUM_KEYPOINTS,
        heatmap_sigma=Config.HEATMAP_SIGMA,
        learning_rate=Config.LEARNING_RATE
    )
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(Config.OUTPUT_DIR, "checkpoints"),
        filename="keypoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    logger = TensorBoardLogger(Config.OUTPUT_DIR, name="logs")
    
    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    print("\nStarting training...")
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s ({training_time/60:.2f} min)")
    
    model_path = os.path.join(Config.OUTPUT_DIR, "final_model.ckpt")
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
        "config": {
            "input_size": Config.INPUT_SIZE,
            "batch_size": Config.BATCH_SIZE,
            "max_epochs": Config.MAX_EPOCHS,
            "learning_rate": Config.LEARNING_RATE,
        },
        "training_time_sec": training_time,
        "metrics": metrics,
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Visualizations saved to: {Config.VIS_DIR}")
    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY!")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    metrics = main()
