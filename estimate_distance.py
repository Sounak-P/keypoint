"""
Document Distance & Camera Height Estimation
=============================================
Estimates camera-to-document distance using A4 paper reference dimensions
and EXIF metadata from images captured on iQOO 9.

Theory:
- Uses pinhole camera model: distance = (focal_length * real_size) / pixel_size
- Extracts EXIF metadata for focal length, sensor info, orientation
- Computes perspective-corrected measurements
"""

import os
import sys
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Some features will be limited.")


# =============================================
# A4 Paper Reference Dimensions
# =============================================
@dataclass
class A4Paper:
    """A4 paper dimensions (ISO 216 standard)"""
    WIDTH_MM: float = 210.0     # Short edge
    HEIGHT_MM: float = 297.0    # Long edge
    WIDTH_CM: float = 21.0
    HEIGHT_CM: float = 29.7
    DIAGONAL_MM: float = 364.0  # sqrt(210^2 + 297^2)
    
    @classmethod
    def get_diagonal_mm(cls) -> float:
        return math.sqrt(cls.WIDTH_MM**2 + cls.HEIGHT_MM**2)


# =============================================
# iQOO 9 Camera Specifications
# =============================================
@dataclass
class IQOO9CameraSpecs:
    """iQOO 9 main camera specifications"""
    # Sensor: Sony IMX598
    SENSOR_NAME: str = "Sony IMX598"
    MEGAPIXELS: float = 48.0
    APERTURE: float = 1.79
    
    # Focal length (35mm equivalent)
    FOCAL_LENGTH_35MM_EQUIV: float = 25.0  # mm
    
    # Actual sensor dimensions (estimated for 1/2" sensor)
    # Sony IMX598 is approximately 1/2" format
    SENSOR_WIDTH_MM: float = 6.4   # Approximate
    SENSOR_HEIGHT_MM: float = 4.8  # Approximate
    
    # Native resolution
    NATIVE_WIDTH: int = 8000
    NATIVE_HEIGHT: int = 6000
    
    # Pixel size (approximate)
    PIXEL_SIZE_UM: float = 0.8  # micrometers
    
    @classmethod
    def get_actual_focal_length_mm(cls) -> float:
        """
        Calculate actual focal length from 35mm equivalent.
        Crop factor = 36mm / sensor_width_mm
        Actual FL = 35mm_equiv_FL / crop_factor
        """
        crop_factor = 36.0 / cls.SENSOR_WIDTH_MM
        actual_fl = cls.FOCAL_LENGTH_35MM_EQUIV / crop_factor
        return actual_fl  # ~4.44mm for iQOO 9


# =============================================
# EXIF Metadata Extraction
# =============================================
def extract_exif_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive EXIF metadata from image.
    """
    metadata = {
        "file": os.path.basename(image_path),
        "exif_available": False,
        "camera": {},
        "image": {},
        "gps": {},
        "orientation": {},
        "raw_exif": {}
    }
    
    try:
        img = Image.open(image_path)
        
        # Basic image info
        metadata["image"]["width"] = img.width
        metadata["image"]["height"] = img.height
        metadata["image"]["format"] = img.format
        metadata["image"]["mode"] = img.mode
        
        # Get EXIF data
        exif_data = img._getexif()
        
        if exif_data:
            metadata["exif_available"] = True
            
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                
                # Store raw tags
                try:
                    if isinstance(value, bytes):
                        metadata["raw_exif"][tag] = value.hex()[:100]
                    else:
                        metadata["raw_exif"][tag] = str(value)[:200]
                except:
                    pass
                
                # Camera info
                if tag == "Make":
                    metadata["camera"]["make"] = str(value)
                elif tag == "Model":
                    metadata["camera"]["model"] = str(value)
                elif tag == "FocalLength":
                    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                        metadata["camera"]["focal_length_mm"] = value.numerator / value.denominator
                    else:
                        metadata["camera"]["focal_length_mm"] = float(value)
                elif tag == "FocalLengthIn35mmFilm":
                    metadata["camera"]["focal_length_35mm"] = int(value)
                elif tag == "FNumber":
                    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                        metadata["camera"]["aperture"] = value.numerator / value.denominator
                    else:
                        metadata["camera"]["aperture"] = float(value)
                elif tag == "ISOSpeedRatings":
                    metadata["camera"]["iso"] = int(value) if isinstance(value, (int, float)) else value
                elif tag == "ExposureTime":
                    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                        metadata["camera"]["exposure_time"] = f"{value.numerator}/{value.denominator}s"
                    else:
                        metadata["camera"]["exposure_time"] = str(value)
                elif tag == "ExifImageWidth":
                    metadata["image"]["exif_width"] = int(value)
                elif tag == "ExifImageHeight":
                    metadata["image"]["exif_height"] = int(value)
                elif tag == "Orientation":
                    orientation_map = {
                        1: "Normal",
                        2: "Mirrored horizontal",
                        3: "Rotated 180",
                        4: "Mirrored vertical",
                        5: "Mirrored horizontal, rotated 90 CCW",
                        6: "Rotated 90 CW",
                        7: "Mirrored horizontal, rotated 90 CW",
                        8: "Rotated 90 CCW"
                    }
                    metadata["orientation"]["value"] = int(value)
                    metadata["orientation"]["description"] = orientation_map.get(int(value), "Unknown")
                elif tag == "DateTimeOriginal":
                    metadata["image"]["datetime"] = str(value)
                elif tag == "ImageDescription":
                    metadata["image"]["description"] = str(value)
                
                # GPS data
                elif tag == "GPSInfo":
                    try:
                        for gps_tag_id, gps_value in value.items():
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            if gps_tag in ["GPSLatitude", "GPSLongitude", "GPSAltitude"]:
                                metadata["gps"][gps_tag] = str(gps_value)
                            elif gps_tag == "GPSLatitudeRef":
                                metadata["gps"]["latitude_ref"] = str(gps_value)
                            elif gps_tag == "GPSLongitudeRef":
                                metadata["gps"]["longitude_ref"] = str(gps_value)
                    except:
                        pass
        
        img.close()
        
    except Exception as e:
        metadata["error"] = str(e)
    
    return metadata


# =============================================
# Distance Calculation
# =============================================
def calculate_pixel_to_mm_ratio(
    keypoints: List[Optional[Tuple[int, int, float]]],
    paper: A4Paper = A4Paper()
) -> Dict[str, float]:
    """
    Calculate the pixel-to-mm ratio using detected document corners.
    
    Uses the known A4 dimensions to determine scaling.
    """
    valid_kps = [kp for kp in keypoints if kp is not None]
    
    if len(valid_kps) < 4:
        return {"error": "Need all 4 corners for accurate measurement"}
    
    # Extract coordinates (assuming order: top_left, top_right, bottom_right, bottom_left)
    tl = np.array([keypoints[0][0], keypoints[0][1]])  # top_left
    tr = np.array([keypoints[1][0], keypoints[1][1]])  # top_right
    br = np.array([keypoints[2][0], keypoints[2][1]])  # bottom_right
    bl = np.array([keypoints[3][0], keypoints[3][1]])  # bottom_left
    
    # Calculate edge lengths in pixels
    top_edge_px = np.linalg.norm(tr - tl)       # Width at top
    bottom_edge_px = np.linalg.norm(br - bl)   # Width at bottom
    left_edge_px = np.linalg.norm(bl - tl)     # Height at left
    right_edge_px = np.linalg.norm(br - tr)    # Height at right
    
    # Average dimensions (accounts for perspective distortion)
    avg_width_px = (top_edge_px + bottom_edge_px) / 2
    avg_height_px = (left_edge_px + right_edge_px) / 2
    
    # Diagonal in pixels
    diagonal1_px = np.linalg.norm(br - tl)
    diagonal2_px = np.linalg.norm(bl - tr)
    avg_diagonal_px = (diagonal1_px + diagonal2_px) / 2
    
    # Determine orientation (portrait vs landscape)
    # In the image, if height > width, the paper appears in portrait
    if avg_height_px > avg_width_px:
        # Paper is in portrait orientation
        width_mm = paper.WIDTH_MM   # 210mm
        height_mm = paper.HEIGHT_MM # 297mm
        orientation = "portrait"
    else:
        # Paper is in landscape orientation
        width_mm = paper.HEIGHT_MM  # 297mm
        height_mm = paper.WIDTH_MM  # 210mm
        orientation = "landscape"
    
    # Calculate pixel-to-mm ratios
    px_per_mm_width = avg_width_px / width_mm
    px_per_mm_height = avg_height_px / height_mm
    px_per_mm_diagonal = avg_diagonal_px / paper.get_diagonal_mm()
    
    # Average ratio (more stable)
    px_per_mm_avg = (px_per_mm_width + px_per_mm_height + px_per_mm_diagonal) / 3
    
    return {
        "orientation": orientation,
        "edges_px": {
            "top": float(top_edge_px),
            "bottom": float(bottom_edge_px),
            "left": float(left_edge_px),
            "right": float(right_edge_px),
            "diagonal_1": float(diagonal1_px),
            "diagonal_2": float(diagonal2_px)
        },
        "average_dimensions_px": {
            "width": float(avg_width_px),
            "height": float(avg_height_px),
            "diagonal": float(avg_diagonal_px)
        },
        "reference_dimensions_mm": {
            "width": width_mm,
            "height": height_mm,
            "diagonal": paper.get_diagonal_mm()
        },
        "px_per_mm": {
            "from_width": float(px_per_mm_width),
            "from_height": float(px_per_mm_height),
            "from_diagonal": float(px_per_mm_diagonal),
            "average": float(px_per_mm_avg)
        },
        "mm_per_px": 1.0 / px_per_mm_avg
    }


def estimate_camera_distance(
    keypoints: List[Optional[Tuple[int, int, float]]],
    image_size: Tuple[int, int],
    metadata: Dict[str, Any],
    camera_specs: IQOO9CameraSpecs = IQOO9CameraSpecs()
) -> Dict[str, Any]:
    """
    Estimate the distance from camera to document using the pinhole camera model.
    
    Formula: distance = (focal_length_mm * real_object_size_mm) / (object_size_px * sensor_size_mm / image_size_px)
    
    Simplified: distance = (focal_length * real_size * image_size) / (object_size_px * sensor_size)
    """
    
    result = {
        "method": "pinhole_camera_model",
        "calculations": {},
        "estimates": {},
        "metadata_used": {}
    }
    
    # Get pixel measurements
    px_info = calculate_pixel_to_mm_ratio(keypoints)
    if "error" in px_info:
        return {"error": px_info["error"]}
    
    result["pixel_measurements"] = px_info
    
    # Get image dimensions
    img_width, img_height = image_size
    
    # Get focal length
    if "focal_length_mm" in metadata.get("camera", {}):
        focal_length_mm = metadata["camera"]["focal_length_mm"]
        result["metadata_used"]["focal_length_source"] = "EXIF"
    else:
        focal_length_mm = camera_specs.get_actual_focal_length_mm()
        result["metadata_used"]["focal_length_source"] = "iQOO 9 specs (estimated)"
    
    result["metadata_used"]["focal_length_mm"] = focal_length_mm
    
    # Sensor dimensions
    # For accurate calculation, we need to account for image resolution vs sensor
    # The sensor gets cropped based on aspect ratio
    if img_width > img_height:
        # Landscape orientation in image sensor
        sensor_width_mm = camera_specs.SENSOR_WIDTH_MM
        sensor_height_mm = camera_specs.SENSOR_HEIGHT_MM
    else:
        # Portrait orientation - sensor is rotated
        sensor_width_mm = camera_specs.SENSOR_HEIGHT_MM
        sensor_height_mm = camera_specs.SENSOR_WIDTH_MM
    
    result["metadata_used"]["sensor_width_mm"] = sensor_width_mm
    result["metadata_used"]["sensor_height_mm"] = sensor_height_mm
    
    # Calculate using multiple methods
    
    # Method 1: Using document width
    if px_info["orientation"] == "portrait":
        real_width_mm = A4Paper.WIDTH_MM
        real_height_mm = A4Paper.HEIGHT_MM
    else:
        real_width_mm = A4Paper.HEIGHT_MM
        real_height_mm = A4Paper.WIDTH_MM
    
    obj_width_px = px_info["average_dimensions_px"]["width"]
    obj_height_px = px_info["average_dimensions_px"]["height"]
    
    # Distance = (focal_length * real_size) / (pixel_size * pixel_pitch)
    # Where pixel_pitch = sensor_size / image_resolution
    
    pixel_pitch_width = sensor_width_mm / img_width
    pixel_pitch_height = sensor_height_mm / img_height
    
    distance_from_width = (focal_length_mm * real_width_mm) / (obj_width_px * pixel_pitch_width)
    distance_from_height = (focal_length_mm * real_height_mm) / (obj_height_px * pixel_pitch_height)
    
    # Method 2: Using diagonal (more stable for perspective)
    real_diagonal_mm = A4Paper.get_diagonal_mm()
    obj_diagonal_px = px_info["average_dimensions_px"]["diagonal"]
    pixel_pitch_diag = math.sqrt(pixel_pitch_width**2 + pixel_pitch_height**2)
    
    # For diagonal, use average pixel pitch
    avg_pixel_pitch = (pixel_pitch_width + pixel_pitch_height) / 2
    distance_from_diagonal = (focal_length_mm * real_diagonal_mm) / (obj_diagonal_px * avg_pixel_pitch)
    
    # Average estimate
    avg_distance_mm = (distance_from_width + distance_from_height + distance_from_diagonal) / 3
    
    result["calculations"] = {
        "pixel_pitch_mm": {
            "horizontal": float(pixel_pitch_width),
            "vertical": float(pixel_pitch_height)
        }
    }
    
    result["estimates"] = {
        "from_width_mm": float(distance_from_width),
        "from_height_mm": float(distance_from_height),
        "from_diagonal_mm": float(distance_from_diagonal),
        "average_mm": float(avg_distance_mm),
        "average_cm": float(avg_distance_mm / 10),
        "average_m": float(avg_distance_mm / 1000),
        "average_inches": float(avg_distance_mm / 25.4)
    }
    
    # Confidence based on consistency of estimates
    estimates = [distance_from_width, distance_from_height, distance_from_diagonal]
    std_dev = np.std(estimates)
    mean_dist = np.mean(estimates)
    cv = (std_dev / mean_dist) * 100 if mean_dist > 0 else 100  # Coefficient of variation
    
    if cv < 5:
        confidence = "high"
    elif cv < 15:
        confidence = "medium"
    else:
        confidence = "low"
    
    result["confidence"] = {
        "level": confidence,
        "coefficient_of_variation_percent": float(cv),
        "std_deviation_mm": float(std_dev)
    }
    
    return result


def estimate_camera_height(
    distance_mm: float,
    keypoints: List[Optional[Tuple[int, int, float]]],
    image_size: Tuple[int, int],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estimate camera height above the document.
    
    This assumes:
    1. Document is lying flat on a surface
    2. Camera is looking down at an angle θ
    3. We can estimate θ from the perspective distortion
    
    If the camera is directly above (θ = 90°), height = distance
    If at an angle, height = distance * sin(θ)
    """
    
    result = {
        "method": "perspective_angle_estimation",
        "assumptions": [
            "Document is flat on horizontal surface",
            "Camera is above the document"
        ]
    }
    
    valid_kps = [kp for kp in keypoints if kp is not None]
    if len(valid_kps) < 4:
        return {"error": "Need all 4 corners"}
    
    # Extract corners
    tl = np.array([keypoints[0][0], keypoints[0][1]])
    tr = np.array([keypoints[1][0], keypoints[1][1]])
    br = np.array([keypoints[2][0], keypoints[2][1]])
    bl = np.array([keypoints[3][0], keypoints[3][1]])
    
    # Calculate perspective indicators
    top_edge = np.linalg.norm(tr - tl)
    bottom_edge = np.linalg.norm(br - bl)
    left_edge = np.linalg.norm(bl - tl)
    right_edge = np.linalg.norm(br - tr)
    
    # Perspective ratio (how much smaller the far edge is)
    # A ratio of 1.0 means overhead view
    width_ratio = min(top_edge, bottom_edge) / max(top_edge, bottom_edge)
    height_ratio = min(left_edge, right_edge) / max(left_edge, right_edge)
    
    result["perspective_analysis"] = {
        "top_edge_px": float(top_edge),
        "bottom_edge_px": float(bottom_edge),
        "left_edge_px": float(left_edge),
        "right_edge_px": float(right_edge),
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio)
    }
    
    # Estimate viewing angle
    # When ratio = 1.0, angle ≈ 90° (overhead)
    # When ratio < 1.0, angle decreases
    # This is a simplified approximation
    
    # More accurate method: Use vanishing point analysis
    # Simplified method: Estimate from perspective distortion
    
    # Average ratio
    avg_ratio = (width_ratio + height_ratio) / 2
    
    # Estimate angle (empirical formula)
    # When ratio = 1.0, sin(angle) = 1 (overhead)
    # When ratio = 0.5, approximately 60° viewing angle
    # sin(θ) ≈ sqrt(ratio) is a reasonable approximation
    
    estimated_sin_angle = math.sqrt(avg_ratio)
    estimated_angle_rad = math.asin(min(estimated_sin_angle, 1.0))
    estimated_angle_deg = math.degrees(estimated_angle_rad)
    
    # Calculate height
    # height = distance * sin(angle)
    camera_height_mm = distance_mm * estimated_sin_angle
    
    # Also calculate horizontal offset
    # horizontal_offset = distance * cos(angle)
    estimated_cos_angle = math.cos(estimated_angle_rad)
    horizontal_offset_mm = distance_mm * estimated_cos_angle
    
    result["viewing_angle"] = {
        "estimated_degrees": float(estimated_angle_deg),
        "sin_angle": float(estimated_sin_angle),
        "cos_angle": float(estimated_cos_angle),
        "note": "90° = directly overhead, lower = more angled view"
    }
    
    result["camera_position"] = {
        "height_mm": float(camera_height_mm),
        "height_cm": float(camera_height_mm / 10),
        "height_m": float(camera_height_mm / 1000),
        "horizontal_offset_mm": float(horizontal_offset_mm),
        "horizontal_offset_cm": float(horizontal_offset_mm / 10)
    }
    
    # Determine confidence
    if avg_ratio > 0.9:
        position_desc = "Camera nearly directly overhead"
        confidence = "high"
    elif avg_ratio > 0.7:
        position_desc = "Camera at moderate angle"
        confidence = "medium"
    else:
        position_desc = "Camera at steep angle - height estimate less reliable"
        confidence = "low"
    
    result["interpretation"] = {
        "description": position_desc,
        "confidence": confidence
    }
    
    return result


def analyze_document_orientation(
    keypoints: List[Optional[Tuple[int, int, float]]],
    image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Analyze how the document is oriented relative to the camera.
    """
    if len([kp for kp in keypoints if kp]) < 4:
        return {"error": "Need all 4 corners"}
    
    tl = np.array([keypoints[0][0], keypoints[0][1]])
    tr = np.array([keypoints[1][0], keypoints[1][1]])
    br = np.array([keypoints[2][0], keypoints[2][1]])
    bl = np.array([keypoints[3][0], keypoints[3][1]])
    
    # Document center
    center = (tl + tr + br + bl) / 4
    image_center = np.array([image_size[0] / 2, image_size[1] / 2])
    
    # Offset from image center
    offset = center - image_center
    
    # Calculate rotation (angle of top edge)
    top_vector = tr - tl
    rotation_rad = math.atan2(top_vector[1], top_vector[0])
    rotation_deg = math.degrees(rotation_rad)
    
    # Calculate document area (using shoelace formula)
    corners = [tl, tr, br, bl]
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2
    
    # Coverage ratio
    image_area = image_size[0] * image_size[1]
    coverage = area / image_area
    
    return {
        "document_center": {
            "x": float(center[0]),
            "y": float(center[1])
        },
        "image_center": {
            "x": float(image_center[0]),
            "y": float(image_center[1])
        },
        "offset_from_center": {
            "x_px": float(offset[0]),
            "y_px": float(offset[1])
        },
        "rotation_degrees": float(rotation_deg),
        "document_area_px2": float(area),
        "image_area_px2": float(image_area),
        "coverage_ratio": float(coverage),
        "coverage_percent": float(coverage * 100)
    }


# =============================================
# Main Analysis
# =============================================
def run_analysis(
    keypoints_file: str,
    image_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Run complete distance and height analysis.
    """
    print("=" * 70)
    print("   DOCUMENT DISTANCE & CAMERA HEIGHT ESTIMATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load keypoints
    print("\n📍 Loading keypoint data...")
    with open(keypoints_file, 'r') as f:
        kp_data = json.load(f)
    
    # Convert to list format
    keypoints = []
    for name in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        kp = kp_data["keypoints"].get(name)
        if kp:
            keypoints.append((kp["x"], kp["y"], kp["confidence"]))
        else:
            keypoints.append(None)
    
    image_size = (kp_data["image"]["width"], kp_data["image"]["height"])
    
    print(f"   Image: {kp_data['image']['file']}")
    print(f"   Size: {image_size[0]} x {image_size[1]}")
    print(f"   Keypoints detected: {sum(1 for kp in keypoints if kp)}/4")
    
    # Extract EXIF metadata
    print("\n📷 Extracting EXIF metadata...")
    metadata = extract_exif_metadata(image_path)
    
    if metadata["exif_available"]:
        print(f"   Camera: {metadata['camera'].get('model', 'Unknown')}")
        print(f"   Focal Length: {metadata['camera'].get('focal_length_mm', 'N/A')} mm")
        print(f"   35mm Equiv: {metadata['camera'].get('focal_length_35mm', 'N/A')} mm")
        print(f"   Aperture: f/{metadata['camera'].get('aperture', 'N/A')}")
        print(f"   ISO: {metadata['camera'].get('iso', 'N/A')}")
    else:
        print("   No EXIF data found - using iQOO 9 default specs")
    
    # Calculate distance
    print("\n📏 Calculating camera distance...")
    distance_result = estimate_camera_distance(
        keypoints, image_size, metadata
    )
    
    if "error" not in distance_result:
        est = distance_result["estimates"]
        print(f"   Distance (from width):    {est['from_width_mm']:.1f} mm")
        print(f"   Distance (from height):   {est['from_height_mm']:.1f} mm")
        print(f"   Distance (from diagonal): {est['from_diagonal_mm']:.1f} mm")
        print(f"   ────────────────────────────────")
        print(f"   📐 Average Distance: {est['average_cm']:.1f} cm ({est['average_mm']:.0f} mm)")
        print(f"   Confidence: {distance_result['confidence']['level']}")
    
    # Calculate camera height
    print("\n📐 Estimating camera height...")
    if "error" not in distance_result:
        height_result = estimate_camera_height(
            distance_result["estimates"]["average_mm"],
            keypoints,
            image_size,
            metadata
        )
        
        if "error" not in height_result:
            pos = height_result["camera_position"]
            angle = height_result["viewing_angle"]
            print(f"   Viewing angle: {angle['estimated_degrees']:.1f}°")
            print(f"   ────────────────────────────────")
            print(f"   📏 Camera Height: {pos['height_cm']:.1f} cm ({pos['height_mm']:.0f} mm)")
            print(f"   Horizontal offset: {pos['horizontal_offset_cm']:.1f} cm")
            print(f"   {height_result['interpretation']['description']}")
    else:
        height_result = {"error": "Distance calculation failed"}
    
    # Analyze document orientation
    print("\n🔄 Analyzing document orientation...")
    orientation_result = analyze_document_orientation(keypoints, image_size)
    
    if "error" not in orientation_result:
        print(f"   Document rotation: {orientation_result['rotation_degrees']:.1f}°")
        print(f"   Image coverage: {orientation_result['coverage_percent']:.1f}%")
    
    # Compile final results
    results = {
        "timestamp": datetime.now().isoformat(),
        "input": {
            "image_file": os.path.basename(image_path),
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "reference": "A4 Paper (210mm x 297mm)"
        },
        "exif_metadata": metadata,
        "keypoints": {
            name: {"x": kp[0], "y": kp[1], "confidence": kp[2]} if kp else None
            for name, kp in zip(
                ["top_left", "top_right", "bottom_right", "bottom_left"],
                keypoints
            )
        },
        "distance_estimation": distance_result,
        "camera_height": height_result if "error" not in distance_result else None,
        "document_orientation": orientation_result,
        "camera_specs_used": {
            "model": "iQOO 9",
            "sensor": IQOO9CameraSpecs.SENSOR_NAME,
            "focal_length_35mm": IQOO9CameraSpecs.FOCAL_LENGTH_35MM_EQUIV,
            "actual_focal_length_mm": IQOO9CameraSpecs.get_actual_focal_length_mm(),
            "sensor_size_mm": f"{IQOO9CameraSpecs.SENSOR_WIDTH_MM} x {IQOO9CameraSpecs.SENSOR_HEIGHT_MM}"
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "distance_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create summary report
    create_visual_report(results, keypoints, image_path, output_dir)
    
    return results


def create_visual_report(
    results: Dict[str, Any],
    keypoints: List[Optional[Tuple[int, int, float]]],
    image_path: str,
    output_dir: str
) -> None:
    """
    Create a visual summary report.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    
    fig = plt.figure(figsize=(16, 12))
    
    # Load and display image
    ax1 = fig.add_subplot(2, 2, 1)
    img = Image.open(image_path)
    ax1.imshow(img)
    
    # Draw keypoints and polygon
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44']
    labels = ['TL', 'TR', 'BR', 'BL']
    
    valid_kps = []
    for i, kp in enumerate(keypoints):
        if kp:
            x, y, conf = kp
            ax1.scatter(x, y, c=colors[i], s=300, marker='o', edgecolors='white', linewidth=2, zorder=5)
            ax1.annotate(labels[i], (x, y), xytext=(15, 15), textcoords='offset points',
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.9))
            valid_kps.append((x, y))
    
    # Draw polygon
    if len(valid_kps) == 4:
        poly = plt.Polygon(valid_kps + [valid_kps[0]], fill=False, 
                          edgecolor='lime', linewidth=3, linestyle='--')
        ax1.add_patch(poly)
    
    ax1.set_title('Detected Document Corners', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Distance visualization
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.axis('off')
    
    if "estimates" in results.get("distance_estimation", {}):
        est = results["distance_estimation"]["estimates"]
        
        # Create info box
        info_text = f"""
╔══════════════════════════════════════════════╗
║      DISTANCE MEASUREMENT RESULTS            ║
╠══════════════════════════════════════════════╣
║                                              ║
║  📏 CAMERA TO DOCUMENT DISTANCE              ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                 ║
║                                              ║
║     {est['average_cm']:.1f} cm  ({est['average_mm']:.0f} mm)              ║
║                                              ║
║  Method breakdown:                           ║
║    • From width:    {est['from_width_mm']:>7.1f} mm             ║
║    • From height:   {est['from_height_mm']:>7.1f} mm             ║
║    • From diagonal: {est['from_diagonal_mm']:>7.1f} mm             ║
║                                              ║
╚══════════════════════════════════════════════╝
"""
        ax2.text(0, 0.5, info_text, transform=ax2.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#4a4a6e'),
                color='#e0e0e0')
    
    ax2.set_title('Distance Estimation', fontsize=14, fontweight='bold')
    
    # Camera height visualization
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.axis('off')
    
    if results.get("camera_height") and "camera_position" in results["camera_height"]:
        pos = results["camera_height"]["camera_position"]
        angle = results["camera_height"]["viewing_angle"]
        interp = results["camera_height"]["interpretation"]
        
        height_text = f"""
╔══════════════════════════════════════════════╗
║        CAMERA HEIGHT ESTIMATION              ║
╠══════════════════════════════════════════════╣
║                                              ║
║  📐 CAMERA HEIGHT ABOVE SURFACE              ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                 ║
║                                              ║
║     {pos['height_cm']:.1f} cm  ({pos['height_mm']:.0f} mm)              ║
║                                              ║
║  Viewing angle: {angle['estimated_degrees']:.1f}°                     ║
║  Horizontal offset: {pos['horizontal_offset_cm']:.1f} cm               ║
║                                              ║
║  Confidence: {interp['confidence'].upper():10s}                   ║
║  {interp['description'][:40]:40s} ║
║                                              ║
╚══════════════════════════════════════════════╝
"""
        ax3.text(0, 0.5, height_text, transform=ax3.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#1a2e1a', edgecolor='#4a6e4a'),
                color='#e0e0e0')
    
    ax3.set_title('Camera Height', fontsize=14, fontweight='bold')
    
    # Metadata and specs
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(-1, 1)
    ax4.axis('off')
    
    meta = results.get("exif_metadata", {})
    cam = meta.get("camera", {})
    specs = results.get("camera_specs_used", {})
    
    meta_text = f"""
╔══════════════════════════════════════════════╗
║           CAMERA & IMAGE METADATA            ║
╠══════════════════════════════════════════════╣
║                                              ║
║  📷 CAMERA INFORMATION                       ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                 ║
║  Camera: {specs.get('model', 'iQOO 9'):20s}           ║
║  Sensor: {specs.get('sensor', 'Sony IMX598'):20s}           ║
║  Focal length: {specs.get('actual_focal_length_mm', 4.44):.2f} mm                  ║
║  35mm equiv: {specs.get('focal_length_35mm', 25):.0f} mm                      ║
║                                              ║
║  📸 FROM EXIF                                ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                 ║
║  Aperture: f/{cam.get('aperture', 'N/A')}                         ║
║  ISO: {str(cam.get('iso', 'N/A')):15s}                    ║
║  Exposure: {str(cam.get('exposure_time', 'N/A')):15s}               ║
║                                              ║
║  📐 REFERENCE                                ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                 ║
║  Document: A4 Paper (210 x 297 mm)           ║
║                                              ║
╚══════════════════════════════════════════════╝
"""
    ax4.text(0, 0.5, meta_text, transform=ax4.transAxes,
            fontsize=11, fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#2e2e1a', edgecolor='#6e6e4a'),
            color='#e0e0e0')
    
    ax4.set_title('Camera Specifications', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    report_path = os.path.join(output_dir, "distance_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Visual report saved to: {report_path}")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    
    # Input files
    keypoints_file = script_dir / "output" / "keypoints_simple.json"
    image_path = script_dir / "IMG_20260207_131354.jpg.jpeg"
    output_dir = script_dir / "output"
    
    # Validate inputs
    if not keypoints_file.exists():
        print(f"ERROR: Keypoints file not found: {keypoints_file}")
        print("Please run infer_keypoints.py first.")
        sys.exit(1)
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    # Run analysis
    results = run_analysis(str(keypoints_file), str(image_path), str(output_dir))
    
    # Print summary
    print("\n" + "=" * 70)
    print("   ANALYSIS COMPLETE")
    print("=" * 70)
    
    if "estimates" in results.get("distance_estimation", {}):
        est = results["distance_estimation"]["estimates"]
        print(f"""
📏 FINAL MEASUREMENTS:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Camera to Document Distance:  {est['average_cm']:.1f} cm
""")
    
    if results.get("camera_height") and "camera_position" in results["camera_height"]:
        pos = results["camera_height"]["camera_position"]
        print(f"""   Camera Height (above surface): {pos['height_cm']:.1f} cm
   Viewing Angle: {results['camera_height']['viewing_angle']['estimated_degrees']:.1f}°
""")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
