"""
Video Keypoint Detection Inference — MULTI-OBJECT
====================================================
Runs UNet keypoint detector on video frames, detects MULTIPLE objects by
extracting all heatmap peaks and grouping them into coherent objects.
Each object gets a stable ID via frame-to-frame Hungarian assignment,
with EMA temporal smoothing and occlusion handling.
"""

import os
import sys
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment


# =============================================
# Configuration
# =============================================
class Config:
    """Inference Configuration"""
    INPUT_SIZE = 256
    NUM_KEYPOINTS = 4
    HEATMAP_SIGMA = 3

    # Inference parameters
    MAX_KEYPOINTS_PER_CHANNEL = 20
    DETECTION_THRESHOLD = 0.1
    MIN_KEYPOINT_DISTANCE = 4

    # Multi-object parameters
    MAX_OBJECTS = 10                 # max objects to detect per frame
    OBJECT_MAX_CORNER_DIST = 400     # max px between any two corners in same object
    MIN_OBJECT_AREA = 100            # minimum quadrilateral area (px²)

    # Keypoint names for paper corners
    KEYPOINT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
    KEYPOINT_SKELETON = [[0, 1], [1, 2], [2, 3], [3, 0]]  # Connect corners

    # Colour palette for object IDs (BGR for OpenCV) – 10 distinct colours
    OBJECT_COLORS_BGR = [
        (72, 61, 255),     # vibrant red-orange
        (87, 227, 137),    # fresh green
        (255, 182, 56),    # electric blue
        (0, 215, 255),     # golden yellow
        (255, 105, 180),   # hot pink
        (50, 205, 50),     # lime green
        (255, 165, 0),     # orange
        (147, 112, 219),   # medium purple
        (0, 255, 255),     # cyan
        (128, 0, 128),     # purple
    ]
    SKELETON_COLOR_BGR = (200, 255, 200)  # light green
    GLOW_COLOR_BGR = (255, 255, 255)      # white glow

    # Temporal smoothing / tracking
    EMA_ALPHA = 0.6              # higher = more weight on current frame
    MAX_MISSING_FRAMES = 3       # frames before an object is considered truly lost
    MATCH_DISTANCE_THRESHOLD = 150  # max centroid distance (px) for frame-to-frame match

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
# Multi-peak extraction (all peaks per channel)
# =============================================
def extract_all_peaks(heatmap_np: np.ndarray) -> List[List[Tuple[int, int, float]]]:
    """
    Return ALL detected peaks per heatmap channel (not just the best one).

    Returns
    -------
    List of length NUM_KEYPOINTS, where each element is a list of
    (x, y, confidence) tuples sorted by confidence descending.
    """
    all_channel_peaks: List[List[Tuple[int, int, float]]] = []

    for ch in range(heatmap_np.shape[0]):
        channel = heatmap_np[ch]
        local_max = maximum_filter(channel, size=Config.MIN_KEYPOINT_DISTANCE * 2 + 1)
        peaks_mask = (channel == local_max) & (channel > Config.DETECTION_THRESHOLD)
        coords = np.where(peaks_mask)

        channel_peaks: List[Tuple[int, int, float]] = []
        if len(coords[0]) > 0:
            vals = channel[coords]
            # Sort by confidence descending, cap at MAX_KEYPOINTS_PER_CHANNEL
            sorted_idx = np.argsort(vals)[::-1][:Config.MAX_KEYPOINTS_PER_CHANNEL]
            for idx in sorted_idx:
                y, x = int(coords[0][idx]), int(coords[1][idx])
                channel_peaks.append((x, y, float(vals[idx])))

        all_channel_peaks.append(channel_peaks)

    return all_channel_peaks


# =============================================
# Group peaks into objects
# =============================================
def _distance(p1: Tuple[int, int, float], p2: Tuple[int, int, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _quad_area(pts: List[Tuple[int, int, float]]) -> float:
    """Shoelace formula for quadrilateral area from 4 (x,y,conf) points."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _is_convex(pts: List[Tuple[int, int, float]]) -> bool:
    """Check if the quadrilateral defined by 4 (x,y,conf) points is roughly convex."""
    n = len(pts)
    if n < 4:
        return False
    signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        signs.append(cross)
    # All same sign → convex
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def group_peaks_into_objects(
    all_peaks: List[List[Tuple[int, int, float]]],
) -> List[List[Optional[Tuple[int, int, float]]]]:
    """
    Group per-channel peaks into coherent objects using greedy nearest-neighbour.

    Strategy:
      1. Seeds come from channel 0 (top_left peaks), sorted by confidence.
      2. For each seed, find the closest unused peak in channels 1, 2, 3 that
         is within OBJECT_MAX_CORNER_DIST of the growing centroid.
      3. Validate the resulting quadrilateral (area, convexity).

    Returns
    -------
    List of objects.  Each object is a list of 4 Optional[(x, y, conf)],
    one per keypoint channel.  Objects are sorted by average confidence desc.
    """
    if not all_peaks or not all_peaks[0]:
        return []

    n_ch = len(all_peaks)
    # Track which peaks have been used in each channel
    used: List[set] = [set() for _ in range(n_ch)]
    objects: List[List[Optional[Tuple[int, int, float]]]] = []

    # Seeds from channel 0, sorted by confidence (they already are)
    for seed_idx, seed in enumerate(all_peaks[0]):
        if seed_idx in used[0]:
            continue
        if len(objects) >= Config.MAX_OBJECTS:
            break

        obj: List[Optional[Tuple[int, int, float]]] = [None] * n_ch
        obj[0] = seed
        used[0].add(seed_idx)

        # Running centroid from assigned corners
        cx, cy = float(seed[0]), float(seed[1])
        n_assigned = 1

        # Greedily assign nearest unused peak in remaining channels
        for ch in range(1, n_ch):
            best_dist = float("inf")
            best_idx = -1
            for p_idx, peak in enumerate(all_peaks[ch]):
                if p_idx in used[ch]:
                    continue
                d = math.hypot(peak[0] - cx, peak[1] - cy)
                if d < best_dist and d < Config.OBJECT_MAX_CORNER_DIST:
                    best_dist = d
                    best_idx = p_idx

            if best_idx >= 0:
                chosen = all_peaks[ch][best_idx]
                obj[ch] = chosen
                used[ch].add(best_idx)
                # Update running centroid
                cx = (cx * n_assigned + chosen[0]) / (n_assigned + 1)
                cy = (cy * n_assigned + chosen[1]) / (n_assigned + 1)
                n_assigned += 1

        # Validate: need at least 3 corners to be useful
        filled = [k for k in obj if k is not None]
        if len(filled) < 3:
            # Release used peaks for this bad object
            for ch_i in range(n_ch):
                if obj[ch_i] is not None:
                    # Find back the index
                    for p_i, p in enumerate(all_peaks[ch_i]):
                        if p is obj[ch_i]:
                            used[ch_i].discard(p_i)
                            break
            continue

        # Optional: area check for 4-corner objects
        if len(filled) == 4:
            area = _quad_area(obj)
            if area < Config.MIN_OBJECT_AREA:
                for ch_i in range(n_ch):
                    if obj[ch_i] is not None:
                        for p_i, p in enumerate(all_peaks[ch_i]):
                            if p is obj[ch_i]:
                                used[ch_i].discard(p_i)
                                break
                continue

        objects.append(obj)

    # Sort by average confidence descending
    def avg_conf(obj):
        confs = [k[2] for k in obj if k is not None]
        return sum(confs) / len(confs) if confs else 0
    objects.sort(key=avg_conf, reverse=True)

    return objects


# =============================================
# Multi-object tracker with stable IDs
# =============================================
@dataclass
class TrackedObject:
    """State for one tracked object."""
    obj_id: int
    # Smoothed keypoints (4 entries, one per channel)
    smooth_kps: List[Optional[Tuple[float, float, float]]] = field(default_factory=lambda: [None]*4)
    missing_frames: int = 0
    # Per-keypoint missing count (for partial occlusion)
    kp_missing: List[int] = field(default_factory=lambda: [0]*4)


class MultiObjectTracker:
    """
    Track multiple objects across frames with stable IDs.

    Uses Hungarian assignment on centroid distances for frame-to-frame matching.
    Each tracked object has EMA-smoothed keypoints and a per-keypoint
    missing-frame counter for handling partial occlusion.
    """

    def __init__(self, alpha: float = 0.6, max_missing: int = 3,
                 match_threshold: float = 150.0):
        self.alpha = alpha
        self.max_missing = max_missing
        self.match_threshold = match_threshold
        self.tracked: OrderedDict[int, TrackedObject] = OrderedDict()
        self._next_id = 1

    def _centroid(self, kps: List[Optional[Tuple]]) -> Optional[Tuple[float, float]]:
        """Compute centroid of non-None keypoints."""
        valid = [(k[0], k[1]) for k in kps if k is not None]
        if not valid:
            return None
        xs, ys = zip(*valid)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def update(
        self,
        detected_objects: List[List[Optional[Tuple[int, int, float]]]],
    ) -> List[Tuple[int, List[Optional[Tuple[int, int, float]]], List[bool]]]:
        """
        Match detected objects to existing tracks, update smoothing, return results.

        Returns
        -------
        List of (obj_id, smoothed_kps, visibility_flags) for each active object.
        """
        # Compute centroids for detections
        det_centroids = []
        for obj in detected_objects:
            c = self._centroid(obj)
            det_centroids.append(c)

        # Compute centroids for existing tracks
        track_ids = list(self.tracked.keys())
        track_centroids = []
        for tid in track_ids:
            t = self.tracked[tid]
            c = self._centroid(t.smooth_kps)
            track_centroids.append(c)

        # Build cost matrix (tracks × detections)
        n_tracks = len(track_ids)
        n_dets = len(detected_objects)

        matched_det = set()
        matched_track = set()

        if n_tracks > 0 and n_dets > 0:
            cost = np.full((n_tracks, n_dets), 1e6)
            for i, tc in enumerate(track_centroids):
                if tc is None:
                    continue
                for j, dc in enumerate(det_centroids):
                    if dc is None:
                        continue
                    cost[i, j] = math.hypot(tc[0] - dc[0], tc[1] - dc[1])

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c_idx in zip(row_ind, col_ind):
                if cost[r, c_idx] < self.match_threshold:
                    matched_track.add(r)
                    matched_det.add(c_idx)
                    tid = track_ids[r]
                    self._update_track(self.tracked[tid], detected_objects[c_idx])

        # Unmatched tracks → increment missing
        for i in range(n_tracks):
            if i not in matched_track:
                tid = track_ids[i]
                self.tracked[tid].missing_frames += 1
                # Increment per-kp missing
                for k in range(Config.NUM_KEYPOINTS):
                    self.tracked[tid].kp_missing[k] += 1

        # Unmatched detections → create new tracks
        for j in range(n_dets):
            if j not in matched_det:
                obj = detected_objects[j]
                new_track = TrackedObject(obj_id=self._next_id)
                self._next_id += 1
                for k in range(Config.NUM_KEYPOINTS):
                    if obj[k] is not None:
                        new_track.smooth_kps[k] = (float(obj[k][0]), float(obj[k][1]), obj[k][2])
                        new_track.kp_missing[k] = 0
                self.tracked[new_track.obj_id] = new_track

        # Remove tracks that have been missing too long
        to_remove = [tid for tid, t in self.tracked.items()
                     if t.missing_frames > self.max_missing]
        for tid in to_remove:
            del self.tracked[tid]

        # Build results
        results: List[Tuple[int, List[Optional[Tuple[int, int, float]]], List[bool]]] = []
        for tid, t in self.tracked.items():
            kps_out: List[Optional[Tuple[int, int, float]]] = []
            vis_out: List[bool] = []
            for k in range(Config.NUM_KEYPOINTS):
                if t.smooth_kps[k] is not None:
                    if t.kp_missing[k] == 0:
                        # Actively detected
                        kps_out.append((int(t.smooth_kps[k][0]),
                                        int(t.smooth_kps[k][1]),
                                        t.smooth_kps[k][2]))
                        vis_out.append(True)
                    elif t.kp_missing[k] <= self.max_missing:
                        # Ghost — recently lost
                        decay = max(0.3, 1.0 - 0.2 * t.kp_missing[k])
                        kps_out.append((int(t.smooth_kps[k][0]),
                                        int(t.smooth_kps[k][1]),
                                        t.smooth_kps[k][2] * decay))
                        vis_out.append(False)
                    else:
                        t.smooth_kps[k] = None
                        kps_out.append(None)
                        vis_out.append(False)
                else:
                    kps_out.append(None)
                    vis_out.append(False)

            results.append((tid, kps_out, vis_out))

        return results

    def _update_track(self, track: TrackedObject, detection: List[Optional[Tuple[int, int, float]]]):
        """EMA-smooth the detection into the track."""
        track.missing_frames = 0
        for k in range(Config.NUM_KEYPOINTS):
            kp = detection[k]
            if kp is not None:
                track.kp_missing[k] = 0
                if track.smooth_kps[k] is None:
                    track.smooth_kps[k] = (float(kp[0]), float(kp[1]), kp[2])
                else:
                    sx = self.alpha * kp[0] + (1 - self.alpha) * track.smooth_kps[k][0]
                    sy = self.alpha * kp[1] + (1 - self.alpha) * track.smooth_kps[k][1]
                    sc = self.alpha * kp[2] + (1 - self.alpha) * track.smooth_kps[k][2]
                    track.smooth_kps[k] = (sx, sy, sc)
            else:
                track.kp_missing[k] += 1


# =============================================
# Per-frame inference (multi-object)
# =============================================
def infer_frame_multiobj(model, frame_rgb: np.ndarray, device: str,
                         orig_w: int, orig_h: int) -> List[List[Optional[Tuple[int, int, float]]]]:
    """
    Run model on a single frame and return a list of detected objects.
    Each object is a list of 4 Optional keypoints scaled to original resolution.
    """
    img = Image.fromarray(frame_rgb).resize((Config.INPUT_SIZE, Config.INPUT_SIZE), Image.BILINEAR)
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmaps = model(tensor)
    hm = heatmaps[0].cpu().numpy()

    # Extract all peaks across all channels
    all_peaks = extract_all_peaks(hm)

    # Group peaks into objects
    objects = group_peaks_into_objects(all_peaks)

    # Scale keypoints to original resolution
    sx, sy = orig_w / Config.INPUT_SIZE, orig_h / Config.INPUT_SIZE
    scaled_objects = []
    for obj in objects:
        scaled_obj = []
        for kp in obj:
            if kp is not None:
                scaled_obj.append((int(kp[0] * sx), int(kp[1] * sy), kp[2]))
            else:
                scaled_obj.append(None)
        scaled_objects.append(scaled_obj)

    return scaled_objects


# =============================================
# Distance estimation (per-object)
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

    if avg_w < 1 or avg_h < 1:
        return None

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
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness)
    _overlay(img, overlay, alpha)


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


def _draw_hud_badge(frame, x, y, title, value, accent_color):
    """Draw a small HUD badge."""
    bw, bh = 150, 55
    _rounded_rect(frame, (x, y), (x + bw, y + bh), Config.HUD_BG, 10, -1, 0.80)
    cv2.rectangle(frame, (x, y), (x + 4, y + bh), accent_color, -1)
    cv2.putText(frame, title, (x + 12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x + 12, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                accent_color, 2, cv2.LINE_AA)


# =============================================
# Multi-object drawing
# =============================================
def draw_multi_object(frame, tracked_objects, frame_idx, fps_val, total_frames):
    """
    Render all tracked objects on frame with per-object colour and ID labels.

    Parameters
    ----------
    tracked_objects : list of (obj_id, kps, visible)
        Output from MultiObjectTracker.update().
    """
    h, w = frame.shape[:2]
    n_objects = len(tracked_objects)
    palette = Config.OBJECT_COLORS_BGR

    for (obj_id, kps, visible) in tracked_objects:
        obj_color = palette[(obj_id - 1) % len(palette)]
        overlay = frame.copy()

        # ── Skeleton lines per object ─────────────────────
        for (i, j) in Config.KEYPOINT_SKELETON:
            if kps[i] is not None and kps[j] is not None:
                p1 = (kps[i][0], kps[i][1])
                p2 = (kps[j][0], kps[j][1])
                if visible[i] and visible[j]:
                    cv2.line(overlay, p1, p2, obj_color, 3, cv2.LINE_AA)
                else:
                    _draw_dashed_line(overlay, p1, p2, (180, 180, 180), 2, dash_len=12)

        _overlay(frame, overlay, 0.7)

        # ── Keypoints with glow ──────────────────────────
        kp_radius = max(8, min(w, h) // 80)
        for i, kp in enumerate(kps):
            if kp is None:
                continue
            cx, cy, conf = kp[0], kp[1], kp[2]

            if visible[i]:
                # Outer glow
                for r_off, a in [(kp_radius + 8, 0.15), (kp_radius + 4, 0.25)]:
                    glow_layer = frame.copy()
                    cv2.circle(glow_layer, (cx, cy), r_off, obj_color, -1, cv2.LINE_AA)
                    _overlay(frame, glow_layer, a)
                # Solid inner
                cv2.circle(frame, (cx, cy), kp_radius, obj_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), kp_radius, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, cy), kp_radius, obj_color, 2, cv2.LINE_AA)

            # Label pill
            label = f"{Config.KEYPOINT_NAMES[i]} {conf:.0%}"
            (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            lx = cx + kp_radius + 6
            ly = cy - kp_radius
            if lx + tw + 10 > w:
                lx = cx - kp_radius - tw - 16
            if ly - th_text - 6 < 0:
                ly = cy + kp_radius + th_text + 10

            _rounded_rect(frame, (lx - 4, ly - th_text - 6), (lx + tw + 6, ly + 6),
                          (20, 20, 20), 6, -1, 0.7)
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        obj_color, 1, cv2.LINE_AA)

        # ── Object ID label near centroid ────────────────
        valid_kps = [k for k in kps if k is not None]
        if valid_kps:
            cent_x = int(sum(k[0] for k in valid_kps) / len(valid_kps))
            cent_y = int(sum(k[1] for k in valid_kps) / len(valid_kps))
            id_label = f"ID {obj_id}"
            (tw, th_text), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
            bx, by = cent_x - tw // 2 - 8, cent_y - th_text // 2 - 8
            _rounded_rect(frame, (bx, by), (bx + tw + 16, by + th_text + 16),
                          obj_color, 8, -1, 0.65)
            cv2.putText(frame, id_label, (bx + 8, by + th_text + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

    # ── HUD: Objects badge (top-left) ──────────────────
    _draw_hud_badge(frame, 15, 15, "OBJECTS",
                    str(n_objects),
                    Config.HUD_ACCENT if n_objects > 0 else Config.HUD_WARN)

    # ── HUD: FPS badge (top-right) ─────────────────────
    fps_text = f"{fps_val:.1f}" if fps_val else "..."
    _draw_hud_badge(frame, w - 165, 15, "FPS", fps_text, (87, 227, 137))

    # ── Per-object distance badges (left side, below OBJECTS badge) ──
    badge_y = 80
    for (obj_id, kps, visible) in tracked_objects:
        dist_cm = estimate_distance_cm(kps, w, h)
        obj_color = palette[(obj_id - 1) % len(palette)]
        dist_str = f"{dist_cm:.1f} cm" if dist_cm is not None else "N/A"
        _draw_hud_badge(frame, 15, badge_y, f"ID {obj_id} DIST", dist_str, obj_color)
        badge_y += 65
        if badge_y > h - 100:
            break  # Don't overflow

    # ── Status bar (bottom) ────────────────────────────
    total_kps = sum(sum(1 for k in kps if k is not None) for _, kps, _ in tracked_objects)
    all_confs = [k[2] for _, kps, _ in tracked_objects for k in kps if k is not None]
    avg_conf = np.mean(all_confs) if all_confs else 0
    status = (f"  Frame {frame_idx + 1}/{total_frames}  |  "
              f"Objects: {n_objects}  |  "
              f"Total Keypoints: {total_kps}  |  "
              f"Avg Conf: {avg_conf:.0%}")

    bar_h = 36
    bar_overlay = frame.copy()
    cv2.rectangle(bar_overlay, (0, h - bar_h), (w, h), Config.HUD_BG, -1)
    _overlay(frame, bar_overlay, 0.75)

    # Accent line
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), Config.HUD_ACCENT, 2)

    cv2.putText(frame, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
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


# =============================================
# Main pipeline
# =============================================
def main():
    print("=" * 70)
    print("   VIDEO KEYPOINT DETECTION — MULTI-OBJECT INFERENCE")
    print("=" * 70)

    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / "toTest/final_model.ckpt"
    video_path = script_dir / "toTest/video_2.mp4"
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "output_video_multiobj.mp4"

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
    print(f"Max objects: {Config.MAX_OBJECTS}")

    # Writer — try mp4v, fall back to XVID
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (vw, vh))
    if not writer.isOpened():
        print("WARN: mp4v codec failed, trying XVID")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_path = output_dir / "output_video_multiobj.avi"
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (vw, vh))

    tracker = MultiObjectTracker(
        alpha=Config.EMA_ALPHA,
        max_missing=Config.MAX_MISSING_FRAMES,
        match_threshold=Config.MATCH_DISTANCE_THRESHOLD,
    )

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

        # Multi-object inference
        raw_objects = infer_frame_multiobj(model, frame_rgb, device, vw, vh)

        # Track across frames (assigns IDs, smooths, handles occlusion)
        tracked = tracker.update(raw_objects)

        # FPS
        dt = time.perf_counter() - t0
        inst_fps = 1.0 / dt if dt > 0 else 0
        fps_ema = 0.3 * inst_fps + 0.7 * fps_ema if fps_ema else inst_fps

        # Draw overlays for all objects
        draw_multi_object(frame, tracked, frame_idx, fps_ema, total_frames)

        writer.write(frame)

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            n_obj = len(tracked)
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx else 0
            print(f"  [{frame_idx}/{total_frames}] - {fps_ema:.1f} fps - "
                  f"{n_obj} object(s) - ETA {eta:.0f}s")

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
