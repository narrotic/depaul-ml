# Broadcast football match

# Camera elevated

# Majority of image is grass

# Grass is green

# No extreme lighting color shifts

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_color(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def show_images(images, titles, cols=4, cmap='gray'):
    """Utility function to display multiple images in a grid layout (from previous homeworks)"""
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(16, 5 * rows))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap=cmap)
        plt.title(title, fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_field_overlay(img_bgr, green_mask, alpha=0.5):
    """
    Overlays a semi-transparent green tint on the original image 
    where the green_mask is positive.
    """
    # Create a solid green image of the same size
    green_overlay = np.zeros_like(img_bgr)
    green_overlay[:] = (0, 255, 0)  # BGR for Green
    
    # Create the blended image: result = img * alpha + overlay * (1 - alpha)
    # We use a copy so we don't mutate the original BGR image
    output = img_bgr.copy()
    
    # We only want to apply the blend where the mask is 255
    mask_indices = green_mask > 0
    
    # Apply the weighted addition only to masked pixels
    output[mask_indices] = cv2.addWeighted(
        img_bgr[mask_indices], alpha, 
        green_overlay[mask_indices], 1 - alpha, 0
    )
    
    return output

# --- Module 1: Preprocessing ---
def preprocess_image(img_bgr):
    """
    Applies Gaussian Blur for noise reduction and returns HSV representation.
    """
    # cv.GaussianBlur(src, kernal_size, sigmaX)
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return blurred, hsv

# --- Module 2: Feature Extraction ---
def extract_features(hsv_img):
    """
    Extracts foreground objects (players) by segmenting out the green field and applying morphological operations.
    """
    h, s, v = cv2.split(hsv_img)
    
    # 1. Segment out the green field
    # Standard OpenCV Hue range for green is approx 35 to 85.
    # For orignal hue value we multiplay by 2 because we are using 180 bins instead of 360.
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Mask of the green field
    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    
    # Invert the mask to get the non-field (foreground features like players, lines)
    fg_mask = cv2.bitwise_not(green_mask)
    
    # 2. Morphological Operations to clean the mask (remove lines, noise)
    # Opening: remove small noise points
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Closing: fill holes within player bodies
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    return mask_closed, green_mask

# --- Module 3: Detection Logic ---
def detect_players(fg_mask, min_area=150, max_area=5000):
    """
    Uses connected components to detect bounding boxes around connected foreground components.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask)
    
    bounding_boxes = []
    
    # Skip the 0th label because it is the background
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by area to exclude tiny noise or massively large connected regions (like stands)
        # Also could filter by aspect ratio since players are usually taller than wide
        aspect_ratio = float(w) / max(h, 1)
        
        if min_area < area < max_area and 0.2 < aspect_ratio < 2.5:
            bounding_boxes.append((x, y, w, h))
            
    return bounding_boxes

def draw_detections(img_bgr, bounding_boxes, color=(0, 255, 0), thickness=2):
    """Draws bounding boxes on an image for visualization."""
    result = img_bgr.copy()
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
    return result

# --- Module 4: Evaluation Utility ---
def compute_iou(boxA, boxB):
    """Computes Intersection over Union between two bounding boxes (x, y, w, h)."""
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    
    # Determine the (x, y)-coordinates of the intersection rectangle
    x_left = max(xA, xB)
    y_top = max(yA, yB)
    x_right = min(xA + wA, xB + wB)
    y_bottom = min(yA + hA, yB + hB)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    boxA_area = wA * hA
    boxB_area = wB * hB
    
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

def evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Evaluates predictions against ground truth boxes.
    Returns Precision, Recall, and F1-Score.
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0
    
    true_positives = 0
    matched_gt = set()
    
    for p_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, g_box in enumerate(gt_boxes):
            if i in matched_gt: continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def parse_ground_truth(json_path, image_name):
    """
    Placeholder utility to parse ground truth bounding boxes for an image.
    Expects format: {"image_name.jpg": [[x, y, w, h], ...]}
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get(image_name, [])
    except FileNotFoundError:
        print(f"Ground truth file not found at {json_path}. Returning empty list.")
        return []

# --- Main Execution ---
def process_image(img_path, gt_json_path=None):
    """Runs the full pipeline on a single image and reports metrics."""
    print(f"\n--- Processing {img_path.name} ---")
    
    # 1. Load & Preprocess
    img_bgr = load_color(img_path)
    blurred, hsv = preprocess_image(img_bgr)
    
    # 2. Extract Features & Overlay
    fg_mask, green_mask = extract_features(hsv)
    overlay_img = create_field_overlay(img_bgr, green_mask, alpha=0.6)
    
    # 3. Detect
    pred_boxes = detect_players(fg_mask)
    result_img = draw_detections(img_bgr, pred_boxes, color=(0, 255, 0))
    
    # 4. Evaluation Logic
    if gt_json_path:
        gt_boxes = parse_ground_truth(gt_json_path, img_path.name)
        
        if gt_boxes:
            # Draw Ground Truth in Red for visual comparison
            result_img = draw_detections(result_img, gt_boxes, color=(0, 0, 255), thickness=1)
            
            # --- QUANTITATIVE RESULTS ---
            p, r, f1 = evaluate_detections(pred_boxes, gt_boxes)
            
            # Print to console
            print(f"Results for {img_path.name}:")
            print(f"  > Precision: {p:.4f}")
            print(f"  > Recall:    {r:.4f}")
            print(f"  > F1-Score:  {f1:.4f}")

            # Burn metrics onto the image (Top-left corner)
            metrics_text = f"P: {p:.2f} | R: {r:.2f} | F1: {f1:.2f}"
            cv2.putText(result_img, metrics_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            print(f"Warning: No ground truth entries found for {img_path.name}")
    else:
        print("Evaluation skipped: No ground truth JSON provided.")

    # 5. Display (Updated for 5 columns)
    show_images(
        [img_bgr, green_mask, overlay_img, fg_mask, result_img],
        ["Original", "Green Mask", "Overlay", "FG Mask", f"Detections ({len(pred_boxes)})"],
        cols=5
    )
    
def main():
    base = Path(__file__).resolve().parents[2] / "data" / "final-project"
    images = [p for p in base.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    gt_json = base / "ground_truth.json"
    
    if not images:
        print(f"No images found in {base}")
        print("Please add dataset images to test the pipeline.")
        return

    print(f"Found {len(images)} image(s) to process.")
    
    for img_path in images:
        process_image(img_path, gt_json_path=gt_json if gt_json.exists() else None)

if __name__ == "__main__":
    main()
