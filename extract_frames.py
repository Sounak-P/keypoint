import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # CVAT 6-digit zero-padded format
        filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(filename, frame)
        
        if frame_idx > 0 and frame_idx % 100 == 0:
            print(f"Extracted {frame_idx} frames...")
            
        frame_idx += 1
        
    cap.release()
    print(f"Done! Extracted {frame_idx} frames to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract all frames from a video as zero-padded PNG images")
    parser.add_argument('--video', type=str, required=True, help="Path to input video")
    parser.add_argument('--out_dir', type=str, required=True, help="Directory to save extracted frames")
    
    args = parser.parse_args()
    extract_frames(args.video, args.out_dir)
