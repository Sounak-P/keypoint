import os
import cv2
import glob
import argparse
from pathlib import Path
from rfdetr import RFDETRBase

def main():
    parser = argparse.ArgumentParser(description="RFDETR inference script for keypoints/objects")
    
    # In Kaggle, the weights are stored in the training output directory. 
    # Usually this will be something like "working/outputs/weights/best.pt"
    # or inside "working/outputs" depending on the save directory structure.
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found!")
        print("Please download your checkpoint from Kaggle. Typically, it is in:")
        print("  <Kaggle Output>/outputs/weights/best.pt (or last.pt)")
        print(f"And then run: python {os.path.basename(__file__)} --weights path/to/best.pt\n")
        return

    print(f"Loading RFDETR model from checkpoint: {args.weights}")
    
    # Initialize and load weights into model
    model = RFDETRBase(args.weights)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Find images
    if os.path.isdir(args.source):
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.source, ext)))
            image_paths.extend(glob.glob(os.path.join(args.source, ext.upper())))
    else:
        # User might pass a single image path
        image_paths = [args.source]
                  
    print(f"Found {len(image_paths)} image(s) to process in '{args.source}'")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        out_path = os.path.join(args.output, img_name)
        
        print(f"Processing {img_name}...")
        
        try:
            # Predict
            results = model.predict(source=img_path)
            
            # Save predictions
            for res in results:
                # Assuming standard YOLO/Ultralytics style result plotting
                annotated_img = res.plot()  # Returns BGR numpy array
                cv2.imwrite(out_path, annotated_img)
                
        except Exception as e:
             # Fallback if standard supervision/plot is slightly different or fails
             print(f"Error visualizing result for {img_name}: {e}")
             # Or if model.predict implicitly saves:
             # model.predict(source=img_path, save=True, project=args.output, name="predictions", exist_ok=True)
                
    print(f"\nInference completed! Check your output folder: {args.output}")

if __name__ == "__main__":
    main()
