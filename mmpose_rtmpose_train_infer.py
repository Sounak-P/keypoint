"""
RTMPose Training & Inference using MMPose/MMCV/MMDet/MMEngine
=============================================================
Uses pre-built mmpose components (CSPNeXt backbone, RTMCCHead, SimCCLabel,
KLDiscretLoss) instead of from-scratch implementations.

"""

import os
import sys
import json
import time
import glob
import math
import warnings
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# =============================================
# 1. Library Validation
# =============================================
def validate_libraries():
    """Check that all required libraries are installed and print versions."""
    print("=" * 60)
    print("LIBRARY VALIDATION")
    print("=" * 60)

    errors = []

    # --- PyTorch ---
    try:
        import torch
        print(f"  ✅ torch           : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"     CUDA available  : Yes ({torch.version.cuda})")
            print(f"     GPU             : {torch.cuda.get_device_name(0)}")
            print(f"     GPU Memory      : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            print("     CUDA available  : No  (will use CPU — training will be slow)")
    except ImportError:
        errors.append("torch is not installed")

    # --- mmengine ---
    try:
        import mmengine
        print(f"  ✅ mmengine        : {mmengine.__version__}")
    except ImportError:
        errors.append("mmengine is not installed. Run: pip install mmengine")

    # --- mmcv ---
    try:
        import mmcv
        print(f"  ✅ mmcv            : {mmcv.__version__}")
    except ImportError:
        errors.append("mmcv is not installed. Run: mim install 'mmcv>=2.0.0'")

    # --- mmdet ---
    try:
        import mmdet
        print(f"  ✅ mmdet           : {mmdet.__version__}")
    except ImportError:
        errors.append("mmdet is not installed. Run: mim install mmdet")

    # --- mmpose ---
    try:
        import mmpose
        print(f"  ✅ mmpose          : {mmpose.__version__}")
    except ImportError:
        errors.append("mmpose is not installed. Run: pip install -e . (from mmpose repo)")

    # --- Verify critical mmpose components ---
    component_checks = {
        'CSPNeXt (backbone)': ('mmdet.models.backbones', 'CSPNeXt'),
        'RTMCCHead':          ('mmpose.models.heads',    'RTMCCHead'),
        'SimCCLabel (codec)': ('mmpose.codecs',          'SimCCLabel'),
        'KLDiscretLoss':      ('mmpose.models.losses',   'KLDiscretLoss'),
        'TopdownPoseEstimator': ('mmpose.models',        'TopdownPoseEstimator'),
    }
    print("\n  Component checks:")
    for name, (module_path, class_name) in component_checks.items():
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name, None)
            if cls is None:
                raise ImportError(f"{class_name} not found in {module_path}")
            print(f"    ✅ {name}")
        except Exception as e:
            print(f"    ❌ {name}: {e}")
            errors.append(f"{name}: {e}")

    # --- Other utilities ---
    for pkg_name in ['xtcocotools', 'scipy']:
        try:
            pkg = __import__(pkg_name)
            ver = getattr(pkg, '__version__', 'OK')
            print(f"  ✅ {pkg_name:16s}: {ver}")
        except ImportError:
            print(f"  ⚠️  {pkg_name:16s}: not found (optional)")

    if errors:
        print("\n❌ VALIDATION FAILED:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)

    print("\n✅ All required libraries validated successfully!")
    print("=" * 60)


# =============================================
# 2. Configuration
# =============================================
class Config:
    """Configuration — mirrors the original script's settings."""

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_rtmpose_mmpose")
    VIS_DIR = os.path.join(BASE_DIR, "outputs_rtmpose_mmpose", "visualizations")
    WORK_DIR = os.path.join(BASE_DIR, "outputs_rtmpose_mmpose", "work_dir")

    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "valid")
    TEST_DIR = os.path.join(DATA_DIR, "test")

    # Model
    INPUT_SIZE = 256
    NUM_KEYPOINTS = 4
    SIMCC_SPLIT_RATIO = 2.0
    SIMCC_SIGMA = 6.0

    # Training
    MAX_EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    NUM_WORKERS = 2
    SEED = 42
    VAL_INTERVAL = 5

    # Inference
    DETECTION_THRESHOLD = 0.1
    DISTANCE_THRESHOLD = 8  # pixels for TP/FP matching

    # Keypoint names
    KEYPOINT_NAMES = ['corner_0', 'corner_1', 'corner_2', 'corner_3']
    SKELETON = []  # no skeleton connections for paper corners
    FLIP_INDICES = [0, 1, 2, 3]

    @classmethod
    def setup(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.VIS_DIR, exist_ok=True)
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        print(f"  Output dir : {cls.OUTPUT_DIR}")
        print(f"  Work dir   : {cls.WORK_DIR}")
        return cls


# =============================================
# 3. YOLO to COCO Annotation Conversion
# =============================================
def yolo_to_coco(data_dir: str, split_name: str, num_keypoints: int = 4) -> str:
    """
    Convert YOLO-format keypoint labels to COCO JSON format.

    YOLO format per line:
        class_id cx cy w h kp0_x kp0_y kp0_v kp1_x kp1_y kp1_v ...
        (all coordinates normalized 0-1)

    Returns path to the generated JSON file.
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    out_json = os.path.join(data_dir, "coco_annotations.json")

    # Collect image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files = sorted(image_files)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "corners",
            "supercategory": "object",
            "keypoints": [f"corner_{i}" for i in range(num_keypoints)],
            "skeleton": [],
        }]
    }

    ann_id = 1
    images_with_annotations = 0

    for img_id, img_path in enumerate(image_files, start=1):
        img = Image.open(img_path)
        img_w, img_h = img.size
        img_filename = os.path.basename(img_path)

        coco["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": img_w,
            "height": img_h,
        })

        # Find corresponding label
        label_name = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            vals = line.strip().split()
            if len(vals) < 5 + num_keypoints * 3:
                continue

            # Parse YOLO bbox (center-x, center-y, w, h) — normalized
            cx = float(vals[1]) * img_w
            cy = float(vals[2]) * img_h
            bw = float(vals[3]) * img_w
            bh = float(vals[4]) * img_h

            # Convert to COCO format [x_min, y_min, width, height]
            x_min = cx - bw / 2
            y_min = cy - bh / 2

            # Parse keypoints
            keypoints_flat = []
            num_visible = 0
            for ki in range(num_keypoints):
                idx = 5 + ki * 3
                kp_x = float(vals[idx]) * img_w
                kp_y = float(vals[idx + 1]) * img_h
                kp_v = int(float(vals[idx + 2]))  # visibility flag
                if kp_v > 0:
                    kp_v = 2  # COCO: 2 = visible
                    num_visible += 1
                keypoints_flat.extend([kp_x, kp_y, kp_v])

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, bw, bh],
                "area": bw * bh,
                "keypoints": keypoints_flat,
                "num_keypoints": num_visible,
                "iscrowd": 0,
            })
            ann_id += 1
            images_with_annotations += 1

    with open(out_json, 'w') as f:
        json.dump(coco, f)

    print(f"  {split_name}: {len(image_files)} images, "
          f"{ann_id - 1} annotations → {out_json}")

    return out_json


def convert_all_splits():
    """Convert all dataset splits from YOLO to COCO format."""
    print("\n" + "-" * 60)
    print("STEP 1: Converting YOLO labels to COCO format")
    print("-" * 60)

    splits = {
        "train": Config.TRAIN_DIR,
        "valid": Config.VAL_DIR,
        "test":  Config.TEST_DIR,
    }
    json_paths = {}
    for name, path in splits.items():
        if os.path.exists(path):
            json_paths[name] = yolo_to_coco(path, name, Config.NUM_KEYPOINTS)
        else:
            print(f"  ⚠️  {name} directory not found: {path}")

    return json_paths


# =============================================
# 4. MMPose Config Builder
# =============================================
def build_mmpose_config(coco_json_paths: Dict[str, str]):
    """
    Load the existing RTMPose-S config from mmpose/projects/rtmpose/
    and override only what differs for our custom 4-keypoint dataset.

    Inherits from the base config:
      ✅ EMA hook (ExpMomentumEMA)
      ✅ Stage-2 pipeline switching (PipelineSwitchHook)
      ✅ Albumentation augmentations (Blur, MedianBlur, CoarseDropout)
      ✅ YOLOXHSVRandomAug
      ✅ RandomHalfBody
      ✅ Gradient clipping (clip_grad)
      ✅ paramwise_cfg (norm/bias decay)
      ✅ auto_scale_lr
    """
    from mmengine.config import Config as MMConfig

    # --- Locate the base RTMPose-S config ---
    base_cfg_path = os.path.join(
        Config.BASE_DIR, 'mmpose', 'projects', 'rtmpose',
        'rtmpose', 'body_2d_keypoint',
        'rtmpose-s_8xb256-420e_coco-256x192.py')

    if not os.path.exists(base_cfg_path):
        raise FileNotFoundError(
            f"RTMPose-S base config not found at: {base_cfg_path}\n"
            f"Make sure the mmpose repo is cloned at {Config.BASE_DIR}/mmpose/")

    print(f"  Loading base config: {base_cfg_path}")
    cfg = MMConfig.fromfile(base_cfg_path)

    # --- Custom metainfo for 4-keypoint paper-corners dataset ---
    custom_metainfo = dict(
        from_config=True,
        dataset_name='paper_corners',
        keypoint_info={
            i: dict(name=f'corner_{i}', id=i, color=[255, 0, 0],
                    type='', swap='')
            for i in range(Config.NUM_KEYPOINTS)
        },
        joint_info={},
        skeleton_info={},
        sigmas=np.array([0.05] * Config.NUM_KEYPOINTS),
    )

    # === Model overrides ===
    # 4 keypoints instead of 17
    cfg.model.head.out_channels = Config.NUM_KEYPOINTS

    # SyncBN → BN for single-GPU training
    cfg.model.backbone.norm_cfg = dict(type='BN')

    # Disable pretrained backbone weights (train from scratch)
    cfg.model.backbone.init_cfg = None

    # Use our SimCC sigma (original uses 4.9/5.66 for COCO proportions)
    input_size = (192, 256)  # (W, H) as in base config
    cfg.codec = dict(
        type='SimCCLabel',
        input_size=input_size,
        sigma=(Config.SIMCC_SIGMA, Config.SIMCC_SIGMA),
        simcc_split_ratio=Config.SIMCC_SPLIT_RATIO,
        normalize=False,
        use_dark=False,
    )
    cfg.model.head.decoder = cfg.codec

    # Disable flip_test (our keypoints don't have symmetric pairs)
    cfg.model.test_cfg = dict(flip_test=False)

    # === Training schedule overrides ===
    cfg.train_cfg.max_epochs = Config.MAX_EPOCHS
    cfg.train_cfg.val_interval = Config.VAL_INTERVAL
    cfg.optim_wrapper.optimizer.lr = Config.LEARNING_RATE

    # Adjust LR scheduler for our epoch count
    cfg.param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=1.0e-5,
            by_epoch=False,
            begin=0,
            end=500),
        dict(
            type='CosineAnnealingLR',
            eta_min=Config.LEARNING_RATE * 0.05,
            begin=Config.MAX_EPOCHS // 2,
            end=Config.MAX_EPOCHS,
            T_max=Config.MAX_EPOCHS // 2,
            by_epoch=True,
            convert_to_iter_based=True),
    ]

    # Adjust stage-2 pipeline switch epoch
    stage2_num_epochs = min(30, Config.MAX_EPOCHS // 4)
    for hook in cfg.custom_hooks:
        if hook.get('type') == 'mmdet.PipelineSwitchHook':
            hook['switch_epoch'] = Config.MAX_EPOCHS - stage2_num_epochs

    # === Dataset overrides ===
    train_ann = coco_json_paths.get('train', '')
    val_ann = coco_json_paths.get('valid', '')
    test_ann = coco_json_paths.get('test', val_ann)

    # Train dataloader
    cfg.train_dataloader.batch_size = Config.BATCH_SIZE
    cfg.train_dataloader.num_workers = Config.NUM_WORKERS
    cfg.train_dataloader.dataset.data_root = Config.TRAIN_DIR
    cfg.train_dataloader.dataset.ann_file = train_ann
    cfg.train_dataloader.dataset.data_prefix = dict(img='images/')
    cfg.train_dataloader.dataset.metainfo = custom_metainfo

    # Val dataloader
    cfg.val_dataloader.batch_size = Config.BATCH_SIZE
    cfg.val_dataloader.num_workers = Config.NUM_WORKERS
    cfg.val_dataloader.dataset.data_root = Config.VAL_DIR
    cfg.val_dataloader.dataset.ann_file = val_ann
    cfg.val_dataloader.dataset.data_prefix = dict(img='images/')
    cfg.val_dataloader.dataset.metainfo = custom_metainfo
    cfg.val_dataloader.dataset.test_mode = True

    # Test dataloader (use test split if available, else val)
    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type='CocoDataset',
            data_root=Config.TEST_DIR,
            data_mode='topdown',
            ann_file=test_ann,
            data_prefix=dict(img='images/'),
            test_mode=True,
            pipeline=cfg.val_dataloader.dataset.pipeline,
            metainfo=custom_metainfo,
        ))

    # Evaluators
    cfg.val_evaluator = dict(
        type='CocoMetric',
        ann_file=val_ann,
        score_mode='bbox',
        nms_mode='none',
    )
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=test_ann,
        score_mode='bbox',
        nms_mode='none',
    )

    # Checkpoint hook
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=Config.VAL_INTERVAL,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=3)

    # === Misc overrides ===
    cfg.randomness = dict(seed=Config.SEED)
    cfg.work_dir = Config.WORK_DIR
    cfg.load_from = None
    cfg.resume = False
    cfg.log_level = 'INFO'

    # Print what we inherited vs overrode
    print(f"  ✅ Inherited from base config:")
    print(f"     - EMA hook (ExpMomentumEMA, momentum=0.0002)")
    print(f"     - Stage-2 pipeline switch (epoch {Config.MAX_EPOCHS - stage2_num_epochs})")
    print(f"     - Albumentation: Blur, MedianBlur, CoarseDropout")
    print(f"     - YOLOXHSVRandomAug, RandomHalfBody")
    print(f"     - Gradient clipping (max_norm=35)")
    print(f"  ✏️  Overridden:")
    print(f"     - num_keypoints: 17 → {Config.NUM_KEYPOINTS}")
    print(f"     - epochs: 420 → {Config.MAX_EPOCHS}")
    print(f"     - batch_size: 256 → {Config.BATCH_SIZE}")
    print(f"     - SyncBN → BN, no pretrained backbone")
    print(f"     - Dataset: COCO → custom paper_corners")

    return cfg


# =============================================
# 5. Training
# =============================================
def train_model(cfg):
    """Train the RTMPose model using mmengine Runner."""
    from mmengine.runner import Runner

    print("\n" + "-" * 60)
    print("STEP 2: Training RTMPose model (mmpose)")
    print("-" * 60)
    print(f"  Backbone       : CSPNeXt-S (deepen=0.33, widen=0.5)")
    print(f"  Head           : RTMCCHead (SimCC, {Config.NUM_KEYPOINTS} keypoints)")
    print(f"  Input size     : {Config.INPUT_SIZE}x{Config.INPUT_SIZE}")
    print(f"  SimCC split    : {Config.SIMCC_SPLIT_RATIO}x")
    print(f"  SimCC sigma    : {Config.SIMCC_SIGMA}")
    print(f"  Batch size     : {Config.BATCH_SIZE}")
    print(f"  Max epochs     : {Config.MAX_EPOCHS}")
    print(f"  Learning rate  : {Config.LEARNING_RATE}")
    print(f"  Optimizer      : AdamW")
    print(f"  Loss           : KLDiscretLoss (beta=10)")

    runner = Runner.from_cfg(cfg)

    # Print model summary
    model = runner.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print(f"  Model size           : ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    print("\n  Starting training...")
    start_time = time.time()
    runner.train()
    training_time = time.time() - start_time
    print(f"\n  Training completed in {training_time:.2f}s ({training_time / 60:.2f} min)")

    return runner, training_time


# =============================================
# 6. Extract Training Logs
# =============================================
def extract_training_curves(work_dir: str) -> Tuple[list, list]:
    """Parse mmengine JSON log files to extract train/val loss curves."""
    train_losses = []
    val_losses = []

    # Find the JSON log file
    log_files = sorted(glob.glob(os.path.join(work_dir, '**', '*.log.json'),
                                 recursive=True))
    if not log_files:
        # Try scalars.json from vis_data
        log_files = sorted(glob.glob(os.path.join(work_dir, '**', 'scalars.json'),
                                     recursive=True))

    if not log_files:
        print("  ⚠️  No log files found for training curves")
        return train_losses, val_losses

    # Parse the latest log file
    log_file = log_files[-1]
    print(f"  Parsing log: {log_file}")

    epoch_train_losses = {}
    epoch_val_losses = {}

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            mode = entry.get('mode', '')
            epoch = entry.get('epoch', None)

            if epoch is None:
                continue

            if mode == 'train' and 'loss' in entry:
                epoch_train_losses[epoch] = entry['loss']
            elif mode == 'val' and 'loss' in entry:
                epoch_val_losses[epoch] = entry['loss']

    # Build ordered lists
    if epoch_train_losses:
        max_epoch = max(epoch_train_losses.keys())
        for e in range(1, max_epoch + 1):
            if e in epoch_train_losses:
                train_losses.append(epoch_train_losses[e])

    if epoch_val_losses:
        max_epoch = max(epoch_val_losses.keys())
        for e in sorted(epoch_val_losses.keys()):
            val_losses.append(epoch_val_losses[e])

    print(f"  Found {len(train_losses)} train epochs, {len(val_losses)} val epochs")
    return train_losses, val_losses


# =============================================
# 7. Inference Engine
# =============================================
class InferenceEngine:
    """Inference with the trained mmpose model + benchmark metrics."""

    def __init__(self, cfg, checkpoint_path: str, device: str = 'cuda'):
        import torch
        from mmengine.config import Config as MMConfig
        from mmengine.runner import Runner

        self.device = device

        # Build model from config
        from mmpose.apis import init_model
        self.model = init_model(cfg, checkpoint_path, device=device)
        self.model.eval()

        self.input_size = Config.INPUT_SIZE
        self.simcc_split_ratio = Config.SIMCC_SPLIT_RATIO

    def predict_single(self, img_path: str) -> Dict:
        """
        Run inference on a single image using mmpose inference API.
        Returns dict with keypoints, scores, and raw SimCC outputs.
        """
        import torch
        from mmpose.apis import inference_topdown
        from mmpose.structures import PoseDataSample

        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # Create a simple bbox covering the whole image (topdown approach)
        bboxes = np.array([[0, 0, img_w, img_h]], dtype=np.float32)

        # Run inference
        results = inference_topdown(self.model, img_path, bboxes=bboxes)

        if results and len(results) > 0:
            result = results[0]
            pred_instances = result.pred_instances

            # keypoints shape: (N, K, 2), scores shape: (N, K)
            keypoints = pred_instances.keypoints[0]  # (K, 2)
            scores = pred_instances.keypoint_scores[0]  # (K,)

            # Scale keypoints to input_size space for comparison
            scale_x = self.input_size / img_w
            scale_y = self.input_size / img_h
            kps_scaled = keypoints.copy()
            kps_scaled[:, 0] *= scale_x
            kps_scaled[:, 1] *= scale_y

            return {
                'keypoints': keypoints.tolist(),
                'keypoints_scaled': kps_scaled.tolist(),
                'scores': scores.tolist(),
                'img_size': (img_w, img_h),
            }
        else:
            return {
                'keypoints': [[0, 0]] * Config.NUM_KEYPOINTS,
                'keypoints_scaled': [[0, 0]] * Config.NUM_KEYPOINTS,
                'scores': [0.0] * Config.NUM_KEYPOINTS,
                'img_size': (img_w, img_h),
            }

    def benchmark(self, test_dir: str, coco_json_path: str,
                  save_visualizations: bool = True) -> Dict:
        """
        Run inference on the test set and compute metrics matching
        the original script's output format.
        """
        import torch

        print(f"\n{'=' * 60}")
        print("RTMPOSE INFERENCE BENCHMARK (mmpose)")
        print("=" * 60)

        # Load COCO annotations for ground truth
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Build image_id → annotations mapping
        img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)

        # Build image_id → image_info mapping
        img_id_to_info = {img['id']: img for img in coco_data['images']}

        images_dir = os.path.join(test_dir, "images")
        num_samples = len(coco_data['images'])
        print(f"  Test images: {num_samples}")

        # Warmup
        print("  Warming up (5 runs)...")
        first_img = coco_data['images'][0]
        warmup_path = os.path.join(images_dir, first_img['file_name'])
        for _ in range(5):
            self.predict_single(warmup_path)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_times = []
        all_predictions = []
        all_ground_truths = []
        all_confidences = []
        memory_usage = []

        print(f"  Benchmarking on {num_samples} samples...")
        for i, img_info in enumerate(coco_data['images']):
            img_path = os.path.join(images_dir, img_info['file_name'])
            img_id = img_info['id']

            if not os.path.exists(img_path):
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            result = self.predict_single(img_path)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            inference_times.append((end - start) * 1000)

            # Predictions in input_size space
            pred_kps = result['keypoints_scaled']
            pred_scores = result['scores']
            all_predictions.append(pred_kps)
            all_confidences.append(pred_scores)

            # Ground truth in input_size space
            gt_kps = []
            anns = img_id_to_anns.get(img_id, [])
            if anns:
                ann = anns[0]
                kps_flat = ann['keypoints']
                img_w = img_info['width']
                img_h = img_info['height']
                for ki in range(Config.NUM_KEYPOINTS):
                    kp_x = kps_flat[ki * 3] * (Config.INPUT_SIZE / img_w)
                    kp_y = kps_flat[ki * 3 + 1] * (Config.INPUT_SIZE / img_h)
                    kp_v = kps_flat[ki * 3 + 2]
                    if kp_v > 0:
                        gt_kps.append([kp_x, kp_y])
                    else:
                        gt_kps.append([-1, -1])
            else:
                gt_kps = [[-1, -1]] * Config.NUM_KEYPOINTS

            all_ground_truths.append(gt_kps)

            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1e6)

            # Visualizations for first 10 images
            if save_visualizations and i < 10:
                self._save_visualization(
                    img_path, gt_kps, pred_kps, pred_scores, i)

            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{num_samples}")

        # Compute metrics
        metrics = self._compute_metrics(
            inference_times, all_predictions, all_ground_truths,
            all_confidences, memory_usage, num_samples)

        return metrics

    def _compute_metrics(self, inference_times, all_predictions,
                         all_ground_truths, all_confidences, memory_usage,
                         num_samples) -> Dict:
        """Compute the same metrics as the original script."""
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
                    dist = np.sqrt((pred[0] - gt[0]) ** 2 +
                                   (pred[1] - gt[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j

                if min_dist <= Config.DISTANCE_THRESHOLD:
                    total_tp += 1
                    distances.append(min_dist)
                    used.add(min_idx)
                else:
                    total_fp += 1

            total_fn += len(valid_gts) - len(used)

        precision = (total_tp / (total_tp + total_fp)
                     if (total_tp + total_fp) > 0 else 0)
        recall = (total_tp / (total_tp + total_fn)
                  if (total_tp + total_fn) > 0 else 0)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        metrics = {
            "model": "RTMPose (mmpose: CSPNeXt + RTMCCHead/SimCC)",
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

    def _save_visualization(self, img_path: str, gt_kps, pred_kps,
                            pred_scores, idx: int):
        """Save GT vs Prediction visualization (3-panel)."""
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img_np = np.array(img)

        scale_x = orig_w / Config.INPUT_SIZE
        scale_y = orig_h / Config.INPUT_SIZE

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # GT
        ax1 = axes[0]
        ax1.imshow(img_np)
        colors_gt = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
        for i, kp in enumerate(gt_kps):
            if kp[0] >= 0 and kp[1] >= 0:
                x, y = kp[0] * scale_x, kp[1] * scale_y
                ax1.scatter(x, y, c=colors_gt[i % 4], s=100, marker='o',
                            edgecolors='white', linewidths=2)
                ax1.annotate(f'GT{i}', (x, y), xytext=(5, 5),
                             textcoords='offset points', color='white',
                             fontsize=8,
                             bbox=dict(boxstyle='round',
                                       facecolor=colors_gt[i % 4], alpha=0.7))
        ax1.set_title('Ground Truth', fontsize=12)
        ax1.axis('off')

        # Predictions
        ax2 = axes[1]
        ax2.imshow(img_np)
        colors_pred = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for i, (kp, conf) in enumerate(zip(pred_kps, pred_scores)):
            x, y = kp[0] * scale_x, kp[1] * scale_y
            ax2.scatter(x, y, c=colors_pred[i % 4], s=80, marker='x',
                        linewidths=2)
            ax2.annotate(f'{conf:.2f}', (x, y), xytext=(5, -10),
                         textcoords='offset points', color='white',
                         fontsize=7,
                         bbox=dict(boxstyle='round',
                                   facecolor=colors_pred[i % 4], alpha=0.7))
        ax2.set_title('RTMPose Predictions (mmpose)', fontsize=12)
        ax2.axis('off')

        # Overlay
        ax3 = axes[2]
        ax3.imshow(img_np)
        for i, kp in enumerate(gt_kps):
            if kp[0] >= 0 and kp[1] >= 0:
                x, y = kp[0] * scale_x, kp[1] * scale_y
                ax3.scatter(x, y, c='lime', s=120, marker='o',
                            edgecolors='white', linewidths=2,
                            label='GT' if i == 0 else '')
        for j, (kp, conf) in enumerate(zip(pred_kps, pred_scores)):
            x, y = kp[0] * scale_x, kp[1] * scale_y
            ax3.scatter(x, y, c='red', s=80, marker='x', linewidths=2,
                        label='Pred' if j == 0 else '')
        ax3.legend(loc='upper right')
        ax3.set_title('Overlay', fontsize=12)
        ax3.axis('off')

        plt.tight_layout()
        vis_path = os.path.join(Config.VIS_DIR, f"pred_{idx:03d}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================
# 8. Visualization Functions
# =============================================
def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    if train_losses:
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        # Val may be at different intervals
        val_epochs = [Config.VAL_INTERVAL * (i + 1)
                      for i in range(len(val_losses))]
        ax1.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (KL-Divergence)', fontsize=12)
    ax1.set_title('RTMPose Training and Validation Loss (mmpose)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if train_losses:
        ax2.plot(range(1, len(train_losses) + 1), train_losses, 'b-',
                 linewidth=1.5, alpha=0.5, label='Train')
        # Smoothed
        window = min(5, len(train_losses))
        if window > 1:
            smoothed = np.convolve(train_losses,
                                   np.ones(window) / window, mode='valid')
            ax2.plot(range(window, len(train_losses) + 1), smoothed, 'b-',
                     linewidth=2, label='Train (smoothed)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Smoothed Training Loss', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved: {save_path}")


def plot_metrics_summary(metrics, save_path):
    """Plot metrics summary — same 2×2 layout as the original script."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Timing
    ax1 = axes[0, 0]
    timing = metrics['timing']
    bars = ax1.bar(
        ['Mean', 'Median', 'P95', 'P99'],
        [timing['mean_ms'], timing['median_ms'],
         timing['p95_ms'], timing['p99_ms']],
        color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('RTMPose Inference Timing (mmpose)', fontsize=14)
    for bar, val in zip(bars, [timing['mean_ms'], timing['median_ms'],
                               timing['p95_ms'], timing['p99_ms']]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, f'{val:.1f}',
                 ha='center', fontsize=10)

    # Detection metrics
    ax2 = axes[0, 1]
    det = metrics['detection']
    bars = ax2.bar(
        ['Precision', 'Recall', 'F1'],
        [det['precision'], det['recall'], det['f1_score']],
        color=['#9b59b6', '#1abc9c', '#34495e'])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Detection Metrics', fontsize=14)
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars, [det['precision'], det['recall'],
                               det['f1_score']]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02, f'{val:.3f}',
                 ha='center', fontsize=10)

    # TP/FP/FN
    ax3 = axes[1, 0]
    labels = ['True Positives', 'False Positives', 'False Negatives']
    values = [det['true_positives'], det['false_positives'],
              det['false_negatives']]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    bars = ax3.bar(labels, values, color=colors)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Detection Breakdown', fontsize=14)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, str(val), ha='center', fontsize=10)

    # Throughput
    ax4 = axes[1, 1]
    th = metrics['throughput']
    ax4.pie([th['fps']], labels=[f"FPS: {th['fps']:.1f}"],
            colors=['#3498db'], autopct='', startangle=90,
            wedgeprops={'width': 0.4})
    ax4.text(0, 0, f"{th['fps']:.1f}\nFPS",
             ha='center', va='center', fontsize=20, fontweight='bold')
    ax4.set_title('Throughput', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Metrics summary saved: {save_path}")


def print_metrics(metrics, training_time=None):
    """Print formatted metrics report — same format as original script."""
    print("\n" + "=" * 60)
    print("RTMPOSE COMPREHENSIVE METRICS REPORT (mmpose)")
    print("=" * 60)
    print(f"Model: {metrics.get('model', 'RTMPose')}")

    if training_time:
        print(f"\n📊 TRAINING: {training_time:.2f}s ({training_time / 60:.2f} min)")

    t = metrics["timing"]
    print(f"\n⏱️  INFERENCE TIMING")
    print(f"   Mean: {t['mean_ms']:.2f}ms | Std: {t['std_ms']:.2f}ms")
    print(f"   Min: {t['min_ms']:.2f}ms | Max: {t['max_ms']:.2f}ms "
          f"| P95: {t['p95_ms']:.2f}ms")

    th = metrics["throughput"]
    print(f"\n🚀 THROUGHPUT: {th['fps']:.2f} FPS")

    m = metrics["memory"]
    print(f"\n💾 MEMORY: Mean {m['mean_mb']:.2f}MB | Max {m['max_mb']:.2f}MB")

    d = metrics["detection"]
    print(f"\n🎯 DETECTION")
    print(f"   Precision: {d['precision']:.4f} | Recall: {d['recall']:.4f} "
          f"| F1: {d['f1_score']:.4f}")
    print(f"   TP: {d['true_positives']} | FP: {d['false_positives']} "
          f"| FN: {d['false_negatives']}")
    print(f"   Mean Error: {d['mean_error_px']:.2f}px")
    print("=" * 60)


# =============================================
# 9. Main
# =============================================
def find_best_checkpoint(work_dir: str) -> str:
    """Find the best checkpoint in the work directory."""
    # Look for best_* checkpoints
    best_ckpts = glob.glob(os.path.join(work_dir, 'best_*.pth'))
    if best_ckpts:
        return sorted(best_ckpts)[-1]

    # Look for the latest epoch checkpoint
    epoch_ckpts = glob.glob(os.path.join(work_dir, 'epoch_*.pth'))
    if epoch_ckpts:
        return sorted(epoch_ckpts)[-1]

    # Look for last_checkpoint file
    last_ckpt_file = os.path.join(work_dir, 'last_checkpoint')
    if os.path.exists(last_ckpt_file):
        with open(last_ckpt_file, 'r') as f:
            return f.read().strip()

    return ""


def main():
    print("=" * 60)
    print("RTMPOSE KEYPOINT DETECTION — TRAINING & INFERENCE")
    print("Using: mmpose / mmcv / mmdet / mmengine")
    print("Architecture: CSPNeXt + RTMCCHead (SimCC)")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 0: Validate libraries
    validate_libraries()

    # Setup
    Config.setup()

    # Step 1: Convert YOLO → COCO
    coco_json_paths = convert_all_splits()

    # Step 2: Build config
    print("\n" + "-" * 60)
    print("Building mmpose config...")
    print("-" * 60)
    cfg = build_mmpose_config(coco_json_paths)

    # Step 3: Train
    runner, training_time = train_model(cfg)

    # Step 4: Extract training curves
    print("\n" + "-" * 60)
    print("STEP 3: Generating training graphs")
    print("-" * 60)
    train_losses, val_losses = extract_training_curves(Config.WORK_DIR)
    if train_losses or val_losses:
        plot_training_curves(
            train_losses, val_losses,
            os.path.join(Config.VIS_DIR, "training_curves.png"))

    # Step 5: Inference benchmark
    print("\n" + "-" * 60)
    print("STEP 4: Inference benchmark with visualizations")
    print("-" * 60)

    # Find best checkpoint
    ckpt_path = find_best_checkpoint(Config.WORK_DIR)
    if not ckpt_path:
        print("  ⚠️  No checkpoint found, using last runner state")
        ckpt_path = os.path.join(Config.WORK_DIR, 'last_checkpoint')
        # Save runner checkpoint as fallback
        import torch
        fallback_path = os.path.join(Config.WORK_DIR, 'final.pth')
        torch.save(runner.model.state_dict(), fallback_path)
        ckpt_path = fallback_path

    print(f"  Using checkpoint: {ckpt_path}")

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    engine = InferenceEngine(cfg, ckpt_path, device)

    test_json = coco_json_paths.get('test',
                                     coco_json_paths.get('valid', ''))
    metrics = engine.benchmark(
        Config.TEST_DIR, test_json, save_visualizations=True)

    # Step 6: Plot metrics & save
    plot_metrics_summary(
        metrics, os.path.join(Config.VIS_DIR, "metrics_summary.png"))

    print_metrics(metrics, training_time)

    # Save results JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "RTMPose (mmpose: CSPNeXt + RTMCCHead/SimCC)",
        "framework": "mmpose + mmcv + mmdet + mmengine",
        "config": {
            "input_size": Config.INPUT_SIZE,
            "batch_size": Config.BATCH_SIZE,
            "max_epochs": Config.MAX_EPOCHS,
            "learning_rate": Config.LEARNING_RATE,
            "simcc_split_ratio": Config.SIMCC_SPLIT_RATIO,
            "simcc_sigma": Config.SIMCC_SIGMA,
            "num_keypoints": Config.NUM_KEYPOINTS,
            "backbone": "CSPNeXt-S (deepen=0.33, widen=0.5)",
            "head": "RTMCCHead",
            "loss": "KLDiscretLoss (beta=10)",
        },
        "training_time_sec": training_time,
        "metrics": metrics,
    }

    metrics_path = os.path.join(Config.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {metrics_path}")

    print(f"\n📁 Visualizations saved to: {Config.VIS_DIR}")
    print("\n" + "=" * 60)
    print("RTMPOSE (MMPOSE) COMPLETED SUCCESSFULLY!")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    metrics = main()
