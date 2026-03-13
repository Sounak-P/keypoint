import os
import cv2
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import supervision as sv
from rfdetr import RFDETRBase


def main():
    parser = argparse.ArgumentParser(description="RFDETR inference script for keypoints/objects")

    parser.add_argument(
        "--weights",
        type=str,
        default=r"c:\TCS\keypoint\toTest\checkpoint_best_total.pth",
        help="Path to the trained RFDETR checkpoint downloaded from Kaggle."
    )

    parser.add_argument(
        "--source",
        type=str,
        default=r"c:\TCS\keypoint\keypointframes\images",
        help="Input path to the images folder or a specific image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=r"c:\TCS\keypoint\output_rfdetr",
        help="Directory to save the prediction results"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )

    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found!")
        print("Please download your checkpoint from Kaggle. Typically, it is in:")
        print("  <Kaggle Output>/outputs/weights/best.pt (or last.pt)")
        print(f"And then run: python {os.path.basename(__file__)} --weights path/to/best.pt\n")
        return

    print(f"Loading RFDETR model from checkpoint: {args.weights}")

    # Initialize model — RFDETRBase() takes no positional args.
    # Use pretrain_weights= to load your custom checkpoint (same pattern as segmentation boilerplate).
    model = RFDETRBase(pretrain_weights=args.weights)

    # Optimise for inference (disables grad, sets eval mode, etc.)
    model.optimize_for_inference()

    os.makedirs(args.output, exist_ok=True)

    # Set up supervision annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Find images
    if os.path.isdir(args.source):
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.source, ext)))
            image_paths.extend(glob.glob(os.path.join(args.source, ext.upper())))
        image_paths = sorted(set(image_paths))
    else:
        image_paths = [args.source]

    print(f"Found {len(image_paths)} image(s) to process in '{args.source}'")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        out_path = os.path.join(args.output, img_name)

        print(f"Processing {img_name}...")

        try:
            # Load image as PIL (required by rfdetr predict)
            img_pil = Image.open(img_path).convert("RGB")

            # Run prediction — returns sv.Detections
            detections: sv.Detections = model.predict(img_pil, threshold=args.threshold)

            # Convert PIL to BGR numpy for OpenCV annotation
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Build label strings (class_id + confidence)
            labels = []
            if detections.class_id is not None and detections.confidence is not None:
                for cls_id, conf in zip(detections.class_id, detections.confidence):
                    labels.append(f"cls{cls_id}: {conf:.2f}")
            elif detections.confidence is not None:
                for conf in detections.confidence:
                    labels.append(f"{conf:.2f}")

            # Annotate image
            annotated = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)
            if labels:
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            cv2.imwrite(out_path, annotated)
            print(f"  -> {len(detections)} detections | saved to {out_path}")

        except Exception as e:
            print(f"  ERROR processing {img_name}: {e}")

    print(f"\nInference completed! Check your output folder: {args.output}")


if __name__ == "__main__":
    main()
