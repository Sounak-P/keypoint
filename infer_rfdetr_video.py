import os
import cv2
import argparse
import numpy as np
from PIL import Image

import supervision as sv
from rfdetr import RFDETRBase


def main():
    parser = argparse.ArgumentParser(description="RFDETR video inference script")

    parser.add_argument(
        "--weights",
        type=str,
        default=r"c:\TCS\keypoint\toTest\checkpoint_best_total.pth",
        help="Path to the trained RFDETR checkpoint."
    )

    parser.add_argument(
        "--source",
        type=str,
        default=r"c:\TCS\keypoint\toTest\video.mp4",
        help="Path to the input video file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=r"c:\TCS\keypoint\output_rfdetr\video_output.mp4",
        help="Path to save the annotated output video."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold."
    )

    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Number of frames to skip between inference calls (0 = process every frame)."
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found!")
        return

    if not os.path.exists(args.source):
        print(f"Error: Video file '{args.source}' not found!")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Loading RFDETR model from: {args.weights}")

    # Load model with custom checkpoint — use pretrain_weights= keyword arg
    model = RFDETRBase(pretrain_weights=args.weights)
    model.optimize_for_inference()

    # Set up supervision annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{args.source}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height} @ {fps:.2f} FPS | {total_frames} frames")
    print(f"Threshold: {args.threshold} | Skip frames: {args.skip_frames}")
    print(f"Output: {args.output}")

    # Set up video writer (mp4v codec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    last_detections = None  # Reuse last detections for skipped frames

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Run inference on this frame (or reuse previous if skipping)
        if args.skip_frames == 0 or frame_idx % (args.skip_frames + 1) == 0:
            try:
                # Convert BGR → RGB PIL for RFDETR
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)

                detections: sv.Detections = model.predict(img_pil, threshold=args.threshold)
                last_detections = detections
            except Exception as e:
                print(f"  Frame {frame_idx}: inference error — {e}")
                last_detections = None
        else:
            detections = last_detections  # Reuse for skipped frames

        # Annotate frame
        annotated = frame_bgr.copy()
        if last_detections is not None and len(last_detections) > 0:
            # Build label strings
            labels = []
            if last_detections.class_id is not None and last_detections.confidence is not None:
                for cls_id, conf in zip(last_detections.class_id, last_detections.confidence):
                    labels.append(f"cls{cls_id}: {conf:.2f}")
            elif last_detections.confidence is not None:
                for conf in last_detections.confidence:
                    labels.append(f"{conf:.2f}")

            annotated = box_annotator.annotate(scene=annotated, detections=last_detections)
            if labels:
                annotated = label_annotator.annotate(scene=annotated, detections=last_detections, labels=labels)

        # Overlay frame counter
        cv2.putText(
            annotated,
            f"Frame {frame_idx}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)

        if frame_idx % 50 == 0:
            det_count = len(last_detections) if last_detections is not None else 0
            print(f"  Frame {frame_idx}/{total_frames} | detections: {det_count}")

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nDone! Processed {frame_idx} frames.")
    print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
