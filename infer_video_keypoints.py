"""
Video Keypoint Detection Inference
====================================
Runs UNet keypoint detector on video frames, overlays keypoints with skeleton
connections, displays estimated distance from camera and real-time FPS.
Handles occlusion / missing corners with temporal smoothing.
"""

import os
import sys
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from PIL import Image
from scipy.ndimage import maximum_filter


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

    # Visually appealing colour palette (BGR for OpenCV)
    KEYPOINT_COLORS_BGR = [
        (72, 61, 255),    # top_left     - vibrant red-orange
        (87, 227, 137),   # top_right    - fresh green
        (255, 182, 56),   # bottom_right - electric blue
        (0, 215, 255),    # bottom_left  - golden yellow
    ]
    SKELETON_COLOR_BGR = (200, 255, 200)  # light green
    GLOW_COLOR_BGR = (255, 255, 255)      # white glow

    # Temporal smoothing
    EMA_ALPHA = 0.6              # higher = more weight on current frame
    MAX_MISSING_FRAMES = 3       # frames before a point is considered truly lost

    # HUD styling
    HUD_BG = (30, 30, 30)
    HUD_ACCENT = (0, 220, 255)
    HUD_TEXT = (240, 240, 240)
    HUD_WARN = (0, 100, 255)


# =============================================
# A4 Paper reference (for distance estimation)
# =============================================
@dataclass
class A4Paper:
    WIDTH_MM: float = 210.0
    HEIGHT_MM: float = 297.0

    @classmethod
    def get_diagonal_mm(cls) -> float:
        return math.sqrt(cls.WIDTH_MM ** 2 + cls.HEIGHT_MM ** 2)


@dataclass
class CameraSpecs:
    """Generic mobile camera specs — tune for your device."""
    FOCAL_LENGTH_35MM_EQUIV: float = 25.0
    SENSOR_WIDTH_MM: float = 6.4
    SENSOR_HEIGHT_MM: float = 4.8

    @classmethod
    def actual_focal_length_mm(cls) -> float:
        crop_factor = 36.0 / cls.SENSOR_WIDTH_MM
        return cls.FOCAL_LENGTH_35MM_EQUIV / crop_factor


# =============================================
# U-Net Backbone (exact copy from training)
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
            nn.ReLU(inplace=True),
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
# Model loading
# =============================================
def load_model(checkpoint_path: str, device: str) -> KeypointDetector:
    print(f"Loading model from: {checkpoint_path}")
    model = KeypointDetector(n_keypoints=Config.NUM_KEYPOINTS, heatmap_sigma=Config.HEATMAP_SIGMA)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.removeprefix("model.")] = v
    model.load_state_dict(new_sd)
    model.to(device).eval()
    print(f"Model loaded on {device}")
    return model


# =============================================
# Per-frame keypoint extraction
# =============================================
def extract_keypoints(heatmap_np: np.ndarray) -> List[Optional[Tuple[int, int, float]]]:
    """Return best (x, y, conf) per channel, or None."""
    best = []
    for ch in range(heatmap_np.shape[0]):
        channel = heatmap_np[ch]
        local_max = maximum_filter(channel, size=Config.MIN_KEYPOINT_DISTANCE * 2 + 1)
        peaks = (channel == local_max) & (channel > Config.DETECTION_THRESHOLD)
        coords = np.where(peaks)
        if len(coords[0]) == 0:
            best.append(None)
            continue
        vals = channel[coords]
        idx = np.argmax(vals)
        y, x = int(coords[0][idx]), int(coords[1][idx])
        best.append((x, y, float(vals[idx])))
    return best


def infer_frame(model, frame_rgb: np.ndarray, device: str, orig_w: int, orig_h: int):
    """Run model on a single frame. Returns scaled keypoints list."""
    img = Image.fromarray(frame_rgb).resize((Config.INPUT_SIZE, Config.INPUT_SIZE), Image.BILINEAR)
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmaps = model(tensor)
    hm = heatmaps[0].cpu().numpy()
    kps = extract_keypoints(hm)

    sx, sy = orig_w / Config.INPUT_SIZE, orig_h / Config.INPUT_SIZE
    scaled = []
    for kp in kps:
        if kp is not None:
            scaled.append((int(kp[0] * sx), int(kp[1] * sy), kp[2]))
        else:
            scaled.append(None)
    return scaled


# =============================================
# Temporal smoothing tracker
# =============================================
class KeypointTracker:
    """EMA smoothing + last-known-position fallback."""
    def __init__(self, n_keypoints: int = 4, alpha: float = 0.6, max_missing: int = 3):
        self.alpha = alpha
        self.max_missing = max_missing
        self.smooth: List[Optional[Tuple[float, float, float]]] = [None] * n_keypoints
        self.missing_count: List[int] = [0] * n_keypoints

    def update(self, raw_kps: List[Optional[Tuple[int, int, float]]]):
        """Update tracker, return smoothed keypoints + visibility flags."""
        out: List[Optional[Tuple[int, int, float]]] = []
        visible: List[bool] = []

        for i, kp in enumerate(raw_kps):
            if kp is not None:
                self.missing_count[i] = 0
                if self.smooth[i] is None:
                    self.smooth[i] = (float(kp[0]), float(kp[1]), kp[2])
                else:
                    sx = self.alpha * kp[0] + (1 - self.alpha) * self.smooth[i][0]
                    sy = self.alpha * kp[1] + (1 - self.alpha) * self.smooth[i][1]
                    sc = self.alpha * kp[2] + (1 - self.alpha) * self.smooth[i][2]
                    self.smooth[i] = (sx, sy, sc)
                out.append((int(self.smooth[i][0]), int(self.smooth[i][1]), self.smooth[i][2]))
                visible.append(True)
            else:
                self.missing_count[i] += 1
                if self.smooth[i] is not None and self.missing_count[i] <= self.max_missing:
                    # Use last known position with decaying confidence
                    decay = max(0.3, 1.0 - 0.2 * self.missing_count[i])
                    out.append((int(self.smooth[i][0]), int(self.smooth[i][1]),
                                self.smooth[i][2] * decay))
                    visible.append(False)  # mark as ghost
                else:
                    self.smooth[i] = None
                    out.append(None)
                    visible.append(False)
        return out, visible


# =============================================
# Distance estimation (simplified A4)
# =============================================
def estimate_distance_cm(kps: List[Optional[Tuple[int, int, float]]],
                         img_w: int, img_h: int) -> Optional[float]:
    """Return estimated camera-to-document distance in cm, or None."""
    valid = [k for k in kps if k is not None]
    if len(valid) < 4:
        return None

    pts = np.array([[k[0], k[1]] for k in kps], dtype=np.float64)
    tl, tr, br, bl = pts

    top_px = np.linalg.norm(tr - tl)
    bot_px = np.linalg.norm(br - bl)
    left_px = np.linalg.norm(bl - tl)
    right_px = np.linalg.norm(br - tr)

    avg_w = (top_px + bot_px) / 2
    avg_h = (left_px + right_px) / 2

    # Determine portrait vs landscape
    if avg_h > avg_w:
        real_w_mm, real_h_mm = A4Paper.WIDTH_MM, A4Paper.HEIGHT_MM
    else:
        real_w_mm, real_h_mm = A4Paper.HEIGHT_MM, A4Paper.WIDTH_MM

    fl = CameraSpecs.actual_focal_length_mm()
    sw = CameraSpecs.SENSOR_WIDTH_MM if img_w > img_h else CameraSpecs.SENSOR_HEIGHT_MM
    sh = CameraSpecs.SENSOR_HEIGHT_MM if img_w > img_h else CameraSpecs.SENSOR_WIDTH_MM

    pp_w = sw / img_w
    pp_h = sh / img_h

    d_w = (fl * real_w_mm) / (avg_w * pp_w)
    d_h = (fl * real_h_mm) / (avg_h * pp_h)
    avg_mm = (d_w + d_h) / 2
    return avg_mm / 10.0  # cm


# =============================================
# Drawing helpers
# =============================================
def _overlay(base, overlay, alpha):
    """Blend overlay onto base with alpha."""
    cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0, base)


def _rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.75):
    """Draw a filled rounded rectangle with transparency."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    # Draw the rounded rectangle using multiple primitives
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness)
    _overlay(img, overlay, alpha)


def draw_keypoints_and_skeleton(frame, kps, visible, frame_idx, fps_val, dist_cm, total_frames):
    """
    Render keypoints, skeleton, HUD on frame (in-place).
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Skeleton lines ──────────────────────────────────
    for (i, j) in Config.KEYPOINT_SKELETON:
        if kps[i] is not None and kps[j] is not None:
            p1 = (kps[i][0], kps[i][1])
            p2 = (kps[j][0], kps[j][1])
            # both visible → solid, any ghost → dashed feel via thinner line + different color
            if visible[i] and visible[j]:
                cv2.line(overlay, p1, p2, Config.SKELETON_COLOR_BGR, 3, cv2.LINE_AA)
            else:
                # Dashed line for ghost segments
                _draw_dashed_line(overlay, p1, p2, (180, 180, 180), 2, dash_len=12)

    _overlay(frame, overlay, 0.7)

    # ── Keypoints with glow ─────────────────────────────
    kp_radius = max(8, min(w, h) // 80)
    for i, kp in enumerate(kps):
        if kp is None:
            continue
        cx, cy, conf = kp[0], kp[1], kp[2]
        color = Config.KEYPOINT_COLORS_BGR[i]

        if visible[i]:
            # Outer glow
            for r_off, a in [(kp_radius + 8, 0.15), (kp_radius + 4, 0.25)]:
                glow_layer = frame.copy()
                cv2.circle(glow_layer, (cx, cy), r_off, color, -1, cv2.LINE_AA)
                _overlay(frame, glow_layer, a)
            # Solid inner
            cv2.circle(frame, (cx, cy), kp_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), kp_radius, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Ghost keypoint (dashed circle effect)
            cv2.circle(frame, (cx, cy), kp_radius, color, 2, cv2.LINE_AA)

        # Label pill
        label = f"{Config.KEYPOINT_NAMES[i]} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx = cx + kp_radius + 6
        ly = cy - kp_radius
        # Keep label inside frame
        if lx + tw + 10 > w:
            lx = cx - kp_radius - tw - 16
        if ly - th - 6 < 0:
            ly = cy + kp_radius + th + 10

        _rounded_rect(frame, (lx - 4, ly - th - 6), (lx + tw + 6, ly + 6), (20, 20, 20), 6, -1, 0.7)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ── HUD: Distance badge (top-left) ─────────────────
    _draw_hud_badge(frame, 15, 15, "DISTANCE",
                    f"{dist_cm:.1f} cm" if dist_cm is not None else "N/A",
                    Config.HUD_ACCENT if dist_cm is not None else Config.HUD_WARN)

    # ── HUD: FPS badge (top-right) ─────────────────────
    fps_text = f"{fps_val:.1f}" if fps_val else "..."
    _draw_hud_badge(frame, w - 165, 15, "FPS", fps_text, (87, 227, 137))

    # ── Status bar (bottom) ────────────────────────────
    detected = sum(1 for k in kps if k is not None)
    avg_conf = np.mean([k[2] for k in kps if k is not None]) if detected else 0
    status = f"  Frame {frame_idx + 1}/{total_frames}  |  Detected: {detected}/4  |  Avg Conf: {avg_conf:.0%}"

    bar_h = 36
    bar_overlay = frame.copy()
    cv2.rectangle(bar_overlay, (0, h - bar_h), (w, h), Config.HUD_BG, -1)
    _overlay(frame, bar_overlay, 0.75)

    # Accent line
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), Config.HUD_ACCENT, 2)

    cv2.putText(frame, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                Config.HUD_TEXT, 1, cv2.LINE_AA)

    # Confidence bar right side
    bar_w_max = 120
    bar_x = w - bar_w_max - 20
    bar_y = h - bar_h + 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + 18), (60, 60, 60), -1)
    fill_w = int(bar_w_max * avg_conf)
    bar_color = (87, 227, 137) if avg_conf > 0.5 else (0, 100, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 18), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + 18), (120, 120, 120), 1)


def _draw_hud_badge(frame, x, y, title, value, accent_color):
    """Draw a small HUD badge."""
    bw, bh = 150, 55
    _rounded_rect(frame, (x, y), (x + bw, y + bh), Config.HUD_BG, 10, -1, 0.80)
    # Accent bar
    cv2.rectangle(frame, (x, y), (x + 4, y + bh), accent_color, -1)
    cv2.putText(frame, title, (x + 12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x + 12, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                accent_color, 2, cv2.LINE_AA)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=10):
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    drawn = 0
    on = True
    while drawn < length:
        seg = min(dash_len, length - drawn)
        sx = int(x1 + ux * drawn)
        sy = int(y1 + uy * drawn)
        ex = int(x1 + ux * (drawn + seg))
        ey = int(y1 + uy * (drawn + seg))
        if on:
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        on = not on
        drawn += seg


# =============================================
# Main pipeline
# =============================================
def main():
    print("=" * 70)
    print("   VIDEO KEYPOINT DETECTION INFERENCE")
    print("=" * 70)

    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / "final_model.ckpt"
    video_path = script_dir / "video.mp4"
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "output_video.mp4"

    for p, label in [(checkpoint_path, "Checkpoint"), (video_path, "Video")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_model(str(checkpoint_path), device)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {vw}x{vh} @ {src_fps:.1f} fps, {total_frames} frames")

    # Writer — try H.264, fall back to XVID
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (vw, vh))
    if not writer.isOpened():
        print("WARN: mp4v codec failed, trying XVID")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_path = output_dir / "output_video.avi"
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (vw, vh))

    tracker = KeypointTracker(
        n_keypoints=Config.NUM_KEYPOINTS,
        alpha=Config.EMA_ALPHA,
        max_missing=Config.MAX_MISSING_FRAMES,
    )

    # Distance EMA
    dist_smooth: Optional[float] = None
    dist_alpha = 0.4

    frame_idx = 0
    t_start = time.perf_counter()
    fps_ema = 0.0

    print(f"\nProcessing {total_frames} frames …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # Convert BGR → RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_kps = infer_frame(model, frame_rgb, device, vw, vh)
        kps, vis = tracker.update(raw_kps)

        # Distance
        dist_cm = estimate_distance_cm(kps, vw, vh)
        if dist_cm is not None:
            dist_smooth = dist_cm if dist_smooth is None else dist_alpha * dist_cm + (1 - dist_alpha) * dist_smooth
        display_dist = dist_smooth

        # FPS
        dt = time.perf_counter() - t0
        inst_fps = 1.0 / dt if dt > 0 else 0
        fps_ema = 0.3 * inst_fps + 0.7 * fps_ema if fps_ema else inst_fps

        # Draw overlays
        draw_keypoints_and_skeleton(frame, kps, vis, frame_idx, fps_ema, display_dist, total_frames)

        writer.write(frame)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx else 0
            print(f"  [{frame_idx}/{total_frames}] - {fps_ema:.1f} fps - ETA {eta:.0f}s")

    cap.release()
    writer.release()

    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE — {frame_idx} frames in {total_time:.1f}s ({frame_idx / total_time:.1f} avg fps)")
    print(f"  Output: {output_path}")
    print(f"  Size:   {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
