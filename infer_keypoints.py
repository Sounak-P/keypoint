"""
Keypoint Detection Inference Script
====================================
Industry-standard inference with COCO-format output and visualizations
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

# =============================================
# Configuration
# =============================================
class Config:
    """Inference Configuration"""
    INPUT_SIZE = 256
    NUM_KEYPOINTS = 4
    HEATMAP_SIGMA = 3
    
    # Inference parameters
    MAX_KEYPOINTS = 20
    DETECTION_THRESHOLD = 0.1
    MIN_KEYPOINT_DISTANCE = 4
    
    # Keypoint names for paper corners
    KEYPOINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
    KEYPOINT_SKELETON = [[0, 1], [1, 2], [2, 3], [3, 0]]  # Connect corners
    
    # Visualization
    KEYPOINT_COLORS = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # RGBA
    SKELETON_COLOR = '#00FF00'


# =============================================
# U-Net Backbone (matching training architecture)
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
class KeypointDetector(nn.Module):
    def __init__(self, n_keypoints=4, heatmap_sigma=3):
        super().__init__()
        
        self.n_keypoints = n_keypoints
        self.heatmap_sigma = heatmap_sigma
        
        self.backbone = UNet(in_channels=3, base_channels=64)
        self.head = nn.Conv2d(self.backbone.out_channels, n_keypoints, kernel_size=1)
    
    def forward(self, x):
        features = self.backbone(x)
        heatmaps = torch.sigmoid(self.head(features))
        return heatmaps


# =============================================
# Keypoint Extraction
# =============================================
def extract_keypoints_from_heatmap(
    heatmap: np.ndarray,
    max_keypoints: int = 20,
    threshold: float = 0.1,
    min_distance: int = 4
) -> Tuple[List[List[Tuple[int, int, float]]], np.ndarray]:
    """
    Extract keypoints from heatmap using non-maximum suppression.
    
    Returns:
        keypoints: List of keypoints per channel, each as (x, y, confidence)
        combined_heatmap: Combined max heatmap for visualization
    """
    from scipy.ndimage import maximum_filter
    
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    keypoints = []
    
    for channel_idx in range(heatmap.shape[0]):
        channel = heatmap[channel_idx]
        channel_kps = []
        
        # Local maxima detection
        local_max = maximum_filter(channel, size=min_distance * 2 + 1)
        peaks = (channel == local_max) & (channel > threshold)
        
        peak_coords = np.where(peaks)
        peak_values = channel[peak_coords]
        
        # Sort by confidence
        sorted_indices = np.argsort(peak_values)[::-1][:max_keypoints]
        
        for idx in sorted_indices:
            y, x = peak_coords[0][idx], peak_coords[1][idx]
            conf = float(peak_values[idx])
            channel_kps.append((int(x), int(y), conf))
        
        keypoints.append(channel_kps)
    
    combined = np.max(heatmap, axis=0)
    return keypoints, combined


def get_best_keypoint_per_channel(
    keypoints: List[List[Tuple[int, int, float]]]
) -> List[Optional[Tuple[int, int, float]]]:
    """Get the highest confidence keypoint from each channel."""
    best_kps = []
    for channel_kps in keypoints:
        if channel_kps:
            best_kps.append(channel_kps[0])  # Already sorted by confidence
        else:
            best_kps.append(None)
    return best_kps


# =============================================
# COCO Format Output
# =============================================
def create_coco_output(
    image_path: str,
    keypoints: List[Optional[Tuple[int, int, float]]],
    image_size: Tuple[int, int],
    inference_time_ms: float
) -> Dict[str, Any]:
    """
    Create COCO-format keypoint annotation output.
    
    COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
    where v = 0 (not labeled), 1 (labeled but not visible), 2 (labeled and visible)
    """
    width, height = image_size
    
    # Flatten keypoints to COCO format
    coco_keypoints = []
    num_visible = 0
    
    for kp in keypoints:
        if kp is not None:
            x, y, conf = kp
            coco_keypoints.extend([int(x), int(y), 2])  # 2 = visible
            num_visible += 1
        else:
            coco_keypoints.extend([0, 0, 0])  # 0 = not labeled
    
    # Calculate bounding box from keypoints
    valid_kps = [kp for kp in keypoints if kp is not None]
    if valid_kps:
        xs = [kp[0] for kp in valid_kps]
        ys = [kp[1] for kp in valid_kps]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    else:
        bbox = [0, 0, width, height]
    
    # COCO annotation format
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "keypoints": coco_keypoints,
        "num_keypoints": num_visible,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        "score": float(np.mean([kp[2] for kp in valid_kps])) if valid_kps else 0.0
    }
    
    # Full COCO output
    coco_output = {
        "info": {
            "description": "Keypoint Detection Inference Output",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [{"id": 1, "name": "Internal Use", "url": ""}],
        "images": [{
            "id": 1,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        }],
        "categories": [{
            "id": 1,
            "name": "document",
            "supercategory": "object",
            "keypoints": Config.KEYPOINT_NAMES,
            "skeleton": Config.KEYPOINT_SKELETON
        }],
        "annotations": [annotation],
        "inference_metadata": {
            "model": "UNet-Keypoint-Detector",
            "inference_time_ms": inference_time_ms,
            "input_size": Config.INPUT_SIZE,
            "detection_threshold": Config.DETECTION_THRESHOLD,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    }
    
    return coco_output


def create_simple_output(
    image_path: str,
    keypoints: List[Optional[Tuple[int, int, float]]],
    image_size: Tuple[int, int],
    inference_time_ms: float
) -> Dict[str, Any]:
    """Create a simplified, human-readable output format."""
    width, height = image_size
    
    keypoint_data = {}
    for i, (name, kp) in enumerate(zip(Config.KEYPOINT_NAMES, keypoints)):
        if kp is not None:
            x, y, conf = kp
            keypoint_data[name] = {
                "x": int(x),
                "y": int(y),
                "confidence": round(conf, 4),
                "normalized_x": round(x / width, 4),
                "normalized_y": round(y / height, 4)
            }
        else:
            keypoint_data[name] = None
    
    return {
        "image": {
            "file": os.path.basename(image_path),
            "width": width,
            "height": height
        },
        "keypoints": keypoint_data,
        "statistics": {
            "detected_count": sum(1 for kp in keypoints if kp is not None),
            "total_expected": len(Config.KEYPOINT_NAMES),
            "average_confidence": round(
                np.mean([kp[2] for kp in keypoints if kp is not None]) if any(kp for kp in keypoints) else 0,
                4
            )
        },
        "inference": {
            "time_ms": round(inference_time_ms, 2),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.now().isoformat()
        }
    }


# =============================================
# Visualization Functions
# =============================================
def visualize_keypoints(
    image: Image.Image,
    keypoints: List[Optional[Tuple[int, int, float]]],
    save_path: str,
    draw_skeleton: bool = True
) -> Image.Image:
    """
    Create visualization with keypoints and skeleton overlay.
    """
    # Create a copy
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    
    # Draw skeleton first (so keypoints are on top)
    if draw_skeleton:
        valid_kps = {i: kp for i, kp in enumerate(keypoints) if kp is not None}
        for start_idx, end_idx in Config.KEYPOINT_SKELETON:
            if start_idx in valid_kps and end_idx in valid_kps:
                x1, y1, _ = valid_kps[start_idx]
                x2, y2, _ = valid_kps[end_idx]
                draw.line([(x1, y1), (x2, y2)], fill=Config.SKELETON_COLOR, width=3)
    
    # Draw keypoints
    radius = max(8, min(image.size) // 50)
    for i, kp in enumerate(keypoints):
        if kp is not None:
            x, y, conf = kp
            color = Config.KEYPOINT_COLORS[i % len(Config.KEYPOINT_COLORS)]
            
            # Draw filled circle
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color,
                outline='white',
                width=2
            )
            
            # Draw label
            label = f"{Config.KEYPOINT_NAMES[i]}\n{conf:.2f}"
            text_x = x + radius + 5
            text_y = y - radius
            
            # Background for text
            draw.rectangle(
                [(text_x - 2, text_y - 2), (text_x + 100, text_y + 30)],
                fill=(0, 0, 0, 180)
            )
            draw.text((text_x, text_y), label, fill='white')
    
    vis_img.save(save_path)
    print(f"Visualization saved: {save_path}")
    return vis_img


def create_detailed_visualization(
    image: Image.Image,
    keypoints: List[Optional[Tuple[int, int, float]]],
    heatmaps: np.ndarray,
    save_path: str
) -> None:
    """
    Create a detailed multi-panel visualization.
    """
    fig = plt.figure(figsize=(20, 10))
    
    # Original image with keypoints
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    
    colors = ['red', 'lime', 'blue', 'yellow']
    markers = ['o', 's', '^', 'd']
    
    for i, kp in enumerate(keypoints):
        if kp is not None:
            x, y, conf = kp
            ax1.scatter(x, y, c=colors[i], s=200, marker=markers[i], 
                       edgecolors='white', linewidths=2, zorder=5)
            ax1.annotate(f'{Config.KEYPOINT_NAMES[i]}\n({conf:.2f})', 
                        (x, y), xytext=(10, 10), textcoords='offset points',
                        fontsize=8, color='white',
                        bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))
    
    # Draw polygon
    valid_kps = [kp for kp in keypoints if kp is not None]
    if len(valid_kps) >= 3:
        polygon = plt.Polygon([(kp[0], kp[1]) for kp in valid_kps], 
                             fill=False, edgecolor='lime', linewidth=2)
        ax1.add_patch(polygon)
    
    ax1.set_title('Detected Keypoints', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Individual heatmaps
    for i in range(min(4, heatmaps.shape[0])):
        ax = fig.add_subplot(2, 3, i + 2)
        im = ax.imshow(heatmaps[i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'{Config.KEYPOINT_NAMES[i]} Heatmap', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Combined heatmap
    ax6 = fig.add_subplot(2, 3, 6)
    combined = np.max(heatmaps, axis=0)
    im = ax6.imshow(combined, cmap='hot', vmin=0, vmax=1)
    ax6.set_title('Combined Heatmap', fontsize=12)
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Detailed visualization saved: {save_path}")


def create_summary_report(
    image: Image.Image,
    keypoints: List[Optional[Tuple[int, int, float]]],
    inference_time: float,
    save_path: str
) -> None:
    """
    Create a professional summary report visualization.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Main image with detections
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44']
    
    for i, kp in enumerate(keypoints):
        if kp is not None:
            x, y, conf = kp
            circle = plt.Circle((x, y), 15, fill=True, color=colors[i], alpha=0.8)
            ax1.add_patch(circle)
            ax1.scatter(x, y, c='white', s=30, marker='+', zorder=10)
    
    # Draw document boundary
    valid_kps = [kp for kp in keypoints if kp is not None]
    if len(valid_kps) == 4:
        # Connect corners in order
        points = [(kp[0], kp[1]) for kp in valid_kps]
        points.append(points[0])  # Close the polygon
        xs, ys = zip(*points)
        ax1.plot(xs, ys, 'g-', linewidth=3, alpha=0.8)
    
    ax1.set_title('Document Corner Detection', fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Statistics panel
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    
    # Create text summary
    detected = sum(1 for kp in keypoints if kp is not None)
    avg_conf = np.mean([kp[2] for kp in keypoints if kp is not None]) if detected > 0 else 0
    
    summary_text = f"""
╔══════════════════════════════════════════════════════╗
║           KEYPOINT DETECTION REPORT                  ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  📊 DETECTION SUMMARY                                ║
║  ─────────────────────                               ║
║  • Keypoints Detected:  {detected} / {len(Config.KEYPOINT_NAMES)}                         ║
║  • Average Confidence:  {avg_conf:.1%}                          ║
║  • Inference Time:      {inference_time:.2f} ms                       ║
║                                                      ║
║  📍 KEYPOINT DETAILS                                 ║
║  ─────────────────────                               ║"""
    
    for i, (name, kp) in enumerate(zip(Config.KEYPOINT_NAMES, keypoints)):
        if kp is not None:
            x, y, conf = kp
            summary_text += f"\n║  • {name:12s}:  ({x:4d}, {y:4d}) @ {conf:.1%}       ║"
        else:
            summary_text += f"\n║  • {name:12s}:  Not detected                 ║"
    
    summary_text += """
║                                                      ║
║  🔧 MODEL INFO                                       ║
║  ─────────────────────                               ║
║  • Architecture:   UNet-Keypoint-Detector            ║
║  • Input Size:     256 x 256                         ║"""
    
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    summary_text += f"\n║  • Device:         {device:30s}║"
    summary_text += "\n╚══════════════════════════════════════════════════════╝"
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#16213e', alpha=0.9),
             color='#e0e0e0')
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=colors[i], label=name) 
                      for i, name in enumerate(Config.KEYPOINT_NAMES)]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Summary report saved: {save_path}")


# =============================================
# Main Inference Function
# =============================================
def load_model(checkpoint_path: str, device: str = 'cuda') -> KeypointDetector:
    """Load model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    model = KeypointDetector(
        n_keypoints=Config.NUM_KEYPOINTS,
        heatmap_sigma=Config.HEATMAP_SIGMA
    )
    
    # Load checkpoint (PyTorch Lightning format)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle PyTorch Lightning checkpoint format
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present (from Lightning)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def run_inference(
    model: KeypointDetector,
    image_path: str,
    device: str = 'cuda'
) -> Tuple[List[Optional[Tuple[int, int, float]]], np.ndarray, float, Tuple[int, int]]:
    """
    Run inference on a single image.
    
    Returns:
        keypoints: List of (x, y, confidence) for each keypoint (in original image coordinates)
        heatmaps: Raw heatmap output
        inference_time_ms: Inference time in milliseconds
        original_size: (width, height) of original image
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    image_resized = image.resize((Config.INPUT_SIZE, Config.INPUT_SIZE), Image.BILINEAR)
    input_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)
    
    # Warmup (first inference is slower)
    with torch.no_grad():
        _ = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed inference
    start_time = time.perf_counter()
    
    with torch.no_grad():
        heatmaps = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Extract keypoints
    heatmaps_np = heatmaps[0].cpu().numpy()
    keypoints_per_channel, _ = extract_keypoints_from_heatmap(
        heatmaps_np,
        max_keypoints=Config.MAX_KEYPOINTS,
        threshold=Config.DETECTION_THRESHOLD,
        min_distance=Config.MIN_KEYPOINT_DISTANCE
    )
    
    # Get best keypoint per channel
    best_keypoints = get_best_keypoint_per_channel(keypoints_per_channel)
    
    # Scale keypoints to original image size
    scale_x = original_size[0] / Config.INPUT_SIZE
    scale_y = original_size[1] / Config.INPUT_SIZE
    
    scaled_keypoints = []
    for kp in best_keypoints:
        if kp is not None:
            x, y, conf = kp
            scaled_keypoints.append((int(x * scale_x), int(y * scale_y), conf))
        else:
            scaled_keypoints.append(None)
    
    return scaled_keypoints, heatmaps_np, inference_time_ms, original_size


def main():
    """Main inference pipeline."""
    print("=" * 70)
    print("   KEYPOINT DETECTION INFERENCE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / "final_model.ckpt"
    image_path = script_dir / "IMG_20260207_131354.jpg.jpeg"
    output_dir = script_dir / "output"
    
    # Validate paths
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print("\n" + "-" * 70)
    print("LOADING MODEL")
    print("-" * 70)
    model = load_model(str(checkpoint_path), device)
    
    # Run inference
    print("\n" + "-" * 70)
    print("RUNNING INFERENCE")
    print("-" * 70)
    keypoints, heatmaps, inference_time, original_size = run_inference(
        model, str(image_path), device
    )
    
    print(f"Inference completed in {inference_time:.2f} ms")
    print(f"Image size: {original_size[0]} x {original_size[1]}")
    
    # Print detected keypoints
    print("\n📍 Detected Keypoints:")
    for i, (name, kp) in enumerate(zip(Config.KEYPOINT_NAMES, keypoints)):
        if kp is not None:
            x, y, conf = kp
            print(f"   {name:12s}: ({x:4d}, {y:4d}) - confidence: {conf:.4f}")
        else:
            print(f"   {name:12s}: Not detected")
    
    # Generate outputs
    print("\n" + "-" * 70)
    print("GENERATING OUTPUTS")
    print("-" * 70)
    
    # Load original image for visualization
    original_image = Image.open(str(image_path)).convert('RGB')
    
    # 1. COCO format JSON
    coco_output = create_coco_output(
        str(image_path), keypoints, original_size, inference_time
    )
    coco_path = output_dir / "keypoints_coco.json"
    with open(coco_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    print(f"✓ COCO format output: {coco_path}")
    
    # 2. Simple JSON format
    simple_output = create_simple_output(
        str(image_path), keypoints, original_size, inference_time
    )
    simple_path = output_dir / "keypoints_simple.json"
    with open(simple_path, 'w') as f:
        json.dump(simple_output, f, indent=2)
    print(f"✓ Simple format output: {simple_path}")
    
    # 3. Basic visualization
    vis_path = output_dir / "visualization.png"
    visualize_keypoints(original_image, keypoints, str(vis_path))
    
    # 4. Detailed visualization with heatmaps
    detailed_path = output_dir / "visualization_detailed.png"
    create_detailed_visualization(original_image, keypoints, heatmaps, str(detailed_path))
    
    # 5. Summary report
    report_path = output_dir / "summary_report.png"
    create_summary_report(original_image, keypoints, inference_time, str(report_path))
    
    # Print summary
    print("\n" + "=" * 70)
    print("   INFERENCE COMPLETE")
    print("=" * 70)
    
    detected = sum(1 for kp in keypoints if kp is not None)
    avg_conf = np.mean([kp[2] for kp in keypoints if kp is not None]) if detected > 0 else 0
    
    print(f"""
📊 Summary:
   • Keypoints detected: {detected} / {len(Config.KEYPOINT_NAMES)}
   • Average confidence: {avg_conf:.2%}
   • Inference time: {inference_time:.2f} ms
   • FPS equivalent: {1000/inference_time:.1f}

📁 Output files:
   • {coco_path}
   • {simple_path}
   • {vis_path}
   • {detailed_path}
   • {report_path}
""")
    
    return keypoints, inference_time


if __name__ == "__main__":
    main()
