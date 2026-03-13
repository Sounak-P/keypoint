"""
Kaggle RFDETR Keypoint Training Script
=======================================
This script prepares the dataset by converting it from the CVAT export
structure into the train/valid/test split structure required by RFDETR,
and then starts the training process.
"""

import os
import json
import shutil
import random
from pathlib import Path

def setup_kaggle_env():
    """Set up the environment paths depending on whether running locally or on Kaggle."""
    if not os.path.exists("/kaggle"):
        print("Running locally")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "keypointframes")
        work_dir = os.path.join(base_dir, "working")
    else:
        print("Running on Kaggle")
        # Update the Kaggle dataset path based on how it's mounted
        # For example, if your dataset is "sounakp/ketpoints"
        data_dir = "/kaggle/input/datasets/sounakp/ketpoints/keypointframes"
        
        # Fallback to other possible paths if the first one doesn't exist
        if not os.path.exists(data_dir):
            data_dir = "/kaggle/input/keypointframes/keypointframes"
        if not os.path.exists(data_dir):
            data_dir = "/kaggle/input/keypointframes"
            
        work_dir = "/kaggle/working"
        
    return data_dir, work_dir

def convert_dataset(data_dir, work_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Converts dataset from:
    ├── annotations/
    │   └── person_keypoints_default.json
    └── images/
        ├── image1.jpg...
        
    To:
    dataset/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── image1.jpg...
    ├── valid/...
    └── test/...
    """
    output_dataset_dir = os.path.join(work_dir, "dataset")
    print(f"Preparing and splitting dataset to {output_dataset_dir} ...")
    
    images_dir = os.path.join(data_dir, "images")
    annotations_file = os.path.join(data_dir, "annotations", "person_keypoints_default.json")
    
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found at: {annotations_file}")
        
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    info = coco_data.get('info', {})
    licenses = coco_data.get('licenses', [])
    
    # Shuffle images securely
    random.seed(42)
    images_copy = images.copy()
    random.shuffle(images_copy)
    
    n_total = len(images_copy)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        'train': images_copy[:n_train],
        'valid': images_copy[n_train:n_train+n_val],
        'test': images_copy[n_train+n_val:]
    }
    
    # Pre-group annotations by image_id for faster lookups
    ann_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)
        
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    for split_name, split_images in splits.items():
        split_dir = os.path.join(output_dataset_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        split_annotations = []
        
        for img in split_images:
            # Copy image
            src_img_path = os.path.join(images_dir, img['file_name'])
            # Only use the basename in case there are subdirectories (like default/) in file_name
            # But COCO file_name usually needs to match exactly in the JSON
            # For robustness, we will keep the exact file_name in JSON, but we have to ensure directories exist
            dst_img_path = os.path.join(split_dir, os.path.basename(img['file_name']))
            
            # Ensure any nested dirs exist if file_name has paths
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image {src_img_path} not found.")
                
            # Also update img['file_name'] to just the basename so RFDETR finds it in the same folder
            img_basename = os.path.basename(img['file_name'])
            img['file_name'] = img_basename
                
            # Get corresponding annotations
            if img['id'] in ann_by_image:
                split_annotations.extend(ann_by_image[img['id']])
                
        # Create COCO JSON struct for this particular split
        split_coco_data = {
            'info': info,
            'licenses': licenses,
            'categories': categories,
            'images': split_images,
            'annotations': split_annotations
        }
        
        split_ann_file = os.path.join(split_dir, "_annotations.coco.json")
        with open(split_ann_file, 'w') as f:
            json.dump(split_coco_data, f, indent=4)
            
        print(f" -> Created '{split_name}' split: {len(split_images)} images, {len(split_annotations)} annotations.")
        
    return output_dataset_dir

def main():
    data_dir, work_dir = setup_kaggle_env()
    
    print(f"Input Data Directory: {data_dir}")
    print(f"Working Directory: {work_dir}")
    
    # 1. Reformat the dataset
    try:
        dataset_dir = convert_dataset(data_dir, work_dir)
    except Exception as e:
        print(f"\nFailed to convert dataset: {e}")
        print("Please check your dataset path and structure.")
        return
        
    # 2. Train the RFDETR model
    print("\nStarting RFDETR Training...")
    
    try:
        from rfdetr import RFDETRBase
    except ImportError:
        print("\nERROR: 'rfdetr' library is not installed.")
        print("Please install it running: pip install rfdetr (or from source) before running this script.")
        return

    # Initialize the base model
    model = RFDETRBase()
    
    # Set up outputs path
    output_path = os.path.join(work_dir, "outputs")
    os.makedirs(output_path, exist_ok=True)

    # Train the model with requested parameters
    print(f"Training on dataset from: {dataset_dir}")
    model.train(
        dataset_dir=dataset_dir,
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=output_path,
        early_stopping=True
    )
    
    print("\nTraining completed successfully!")
    print(f"Outputs are saved to: {output_path}")

if __name__ == "__main__":
    main()
