import cv2
import json
import argparse
import os
import re

def main(args):
    print(f"Loading annotations from {args.json_path}")
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        return

    with open(args.json_path, 'r') as f:
        coco_data = json.load(f)
        
    # Map image id to annotations
    image_id_to_anns = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in image_id_to_anns:
            image_id_to_anns[image_id] = []
        image_id_to_anns[image_id].append(ann)
        
    print(f"Loaded {len(coco_data.get('annotations', []))} annotations for {len(image_id_to_anns)} images.")

    # Map frame index to image_id
    # CVAT typically names frames like 'frame_000000.png'
    frame_idx_to_image_id = {}
    for img in coco_data.get('images', []):
        filename = img['file_name']
        basename = os.path.basename(filename)
        # Find all number sequences in the filename
        numbers = re.findall(r'\d+', basename)
        if numbers:
            # Usually the frame number is the last number sequence before the extension
            frame_idx = int(numbers[-1])
            frame_idx_to_image_id[frame_idx] = img['id']

    # Open video
    print(f"Opening video {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Default to 30 FPS if couldn't read from video
    if fps <= 0:
        fps = 30.0
        
    out = None
    if args.output:
        # Setup VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        
    print(f"Video props: {width}x{height} @ {fps}fps, Total frames: {total_frames}")

    # Extract keypoint labels if they exist
    categories = coco_data.get('categories', [])
    keypoint_names = []
    if categories and 'keypoints' in categories[0]:
        keypoint_names = categories[0]['keypoints']
    
    frame_idx = 0
    frames_processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw annotations
        image_id = frame_idx_to_image_id.get(frame_idx)
        if image_id is not None and image_id in image_id_to_anns:
            anns = image_id_to_anns[image_id]
            for ann in anns:
                # Draw bounding box
                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                    
                # Draw keypoints
                if 'keypoints' in ann:
                    kpts = ann['keypoints']
                    # CVAT COCO format: list of [x, y, visibility]
                    # visibility: 0: not labeled, 1: labeled but not visible, 2: labeled and visible
                    for i in range(0, len(kpts), 3):
                        kx, ky, kv = kpts[i], kpts[i+1], kpts[i+2]
                        if kv > 0: 
                            kpt_name = keypoint_names[i//3] if i//3 < len(keypoint_names) else str(i//3)
                            
                            # Color: Red if not visible (kv==1), Blue if visible (kv==2)
                            color = (0, 0, 255) if kv == 1 else (255, 0, 0)
                            
                            # Draw point
                            cv2.circle(frame, (int(kx), int(ky)), 4, color, -1)
                            # Draw name label
                            cv2.putText(frame, kpt_name, (int(kx)+5, int(ky)-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if out is not None:
            out.write(frame)
            
        if args.out_dir:
            out_img_path = os.path.join(args.out_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(out_img_path, frame)
            
        frame_idx += 1
        frames_processed += 1
        
        if frames_processed % 100 == 0:
            print(f"Processed {frames_processed} frames...")
            
    cap.release()
    if out is not None:
        out.release()
    print(f"Done! Processed {frames_processed} total frames.")
    if args.output:
        print(f"Output saved to video: {args.output}")
    if args.out_dir:
        print(f"Output frames saved to dir: {args.out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Overlay CVAT COCO annotations on video")
    parser.add_argument('--video', type=str, required=True, help="Path to input video")
    parser.add_argument('--json_path', type=str, 
                        default=r'c:\TCS\keypoint\allframes-annotation\annotations\person_keypoints_default.json', 
                        help="Path to COCO JSON annotation file")
    parser.add_argument('--output', type=str, default='output_overlay.mp4', help="Path to output video (set to empty to skip video generation)")
    parser.add_argument('--out_dir', type=str, default='', help="Directory to save individual annotated frames as images")
    
    args = parser.parse_args()
    main(args)
