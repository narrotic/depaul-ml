import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def show_images_in_row(images, titles, figsize=(15, 5)):
    """Helper function to display a list of images in a row."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        # Convert to RGB if it's a color image
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def part1_locating_blue_piece(img_bgr):
    """
    Part 1: Locate the blue piece using color segmentation (HSV)
    and morphological operations.
    Returns:
        blue_mask: Binary mask of the blue piece.
        visualized: Image with all other pixels' intensity lowered.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Define hue range for blue pixels. Note that OpenCV H is 0-179.
    # We use a broad range to combat illumination changes.
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    
    # Threshold HSV to get initial mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Morphological operations to recover missing pixels and remove noise
    # We use a large ellipse kernel as recommended
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    
    # Closing to fill holes inside the blue piece
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Opening to remove noisy pixels outside the blue piece
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Explicit analysis of connected components in case of remaining large noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        # label 0 is background, so we want the largest component from label 1 onwards
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        blue_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        blue_mask = mask.copy()

    # Visualize by lowering intensity of all other pixels
    visualized = img_bgr.copy()
    # Reduce intensity of background (multiply by 0.3)
    visualized = (visualized * 0.3).astype(np.uint8)
    # Restore the blue piece back to full intensity
    visualized[blue_mask == 255] = img_bgr[blue_mask == 255]
    
    return blue_mask, visualized

def part2_locating_boundaries(img_bgr, blue_mask):
    """
    Part 2: Locate the boundaries of the shapes.
    Expected output has white background and black boundaries.
    Returns:
        refined_boundaries: Binary image of boundaries.
        boundaries_no_blue: Binary image after removing blue piece pixels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Binarize with Otsu, inverted so dark ink becomes white (255)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operation to close small gaps in the shape outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    refined_boundaries = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    
    # Remove pixels of the blue piece
    # We slightly dilate the blue mask to ensure complete removal of its edges 
    # from the boundaries mask if they overlapped.
    removal_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_blue = cv2.dilate(blue_mask, removal_kernel, iterations=1)
    
    boundaries_no_blue = refined_boundaries.copy()
    boundaries_no_blue[dilated_blue == 255] = 0
    
    # Invert to match assignment requirements (white background, black boundary)
    refined_boundaries = cv2.bitwise_not(refined_boundaries)
    boundaries_no_blue = cv2.bitwise_not(boundaries_no_blue)
    
    return refined_boundaries, boundaries_no_blue

def part3_locating_target_shape(img_bgr, boundaries_no_blue, blue_mask):
    """
    Part 3: Locate target shape using CC analysis.
    Returns:
        target_cc_mask: Binary mask of the CC that contains the blue piece.
        highlighted: Original image with the shape colored to highlight it.
    """
    # Since boundaries_no_blue has white background and black boundaries,
    # the interiors of the shapes will be white areas separated by black boundaries.
    interiors = boundaries_no_blue.copy()
    
    # We need to perform CC on the white regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interiors)
    
    best_overlap = 0
    target_label = -1
    
    # Check all labels to find which has maximum overlap with blue piece.
    for label in range(0, num_labels):
        cc_mask = (labels == label).astype(np.uint8) * 255
        
        # Overlap with the blue mask
        overlap = cv2.bitwise_and(cc_mask, blue_mask)
        overlap_area = cv2.countNonZero(overlap)
        
        if overlap_area > best_overlap:
            best_overlap = overlap_area
            target_label = label
            
    # Reconstruct the target mask
    if target_label != -1:
        target_cc_mask = (labels == target_label).astype(np.uint8) * 255
    else:
        # Fallback if somehow no overlap found
        target_cc_mask = np.zeros_like(blue_mask)

    # Visualize target shape by changing its color
    highlighted = img_bgr.copy()
    
    # Create an overlay with a yellow tint (BGR: 0, 255, 255)
    overlay = highlighted.copy()
    overlay[target_cc_mask == 255] = [0, 255, 255]
    
    # Blend the overlay with the original image inside the target mask
    # We use a mask so we only blend within the target shape
    alpha = 0.5
    blended = cv2.addWeighted(overlay, alpha, highlighted, 1 - alpha, 0)
    
    # Apply blended region back to highlighted image
    highlighted[target_cc_mask == 255] = blended[target_cc_mask == 255]

    return target_cc_mask, highlighted

def main():
    base = Path(__file__).resolve().parents[2] / "data" / "hw7"
    
    images = [
        p for p in base.glob("*.*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    
    if not images:
        print(f"No images found in {base}")
        return
        
    print(f"Found {len(images)} images to process.")
    
    for img_path in sorted(images):
        print(f"\n--- Processing: {img_path.name} ---")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Could not read {img_path.name}, skipping.")
            continue
            
        # Part 1: Locating the Blue Piece
        blue_mask, visualized_blue = part1_locating_blue_piece(img_bgr)
        show_images_in_row(
            [blue_mask, visualized_blue],
            ["Part 1.a: Blue Piece Mask", "Part 1.b: Highlighted Blue Piece"]
        )
        
        # Part 2: Locating the Boundaries of the Shapes
        refined_boundaries, boundaries_no_blue = part2_locating_boundaries(img_bgr, blue_mask)
        show_images_in_row(
            [refined_boundaries, boundaries_no_blue],
            ["Part 2.a: Refined Boundaries", "Part 2.b: Boundaries (No Blue Piece)"]
        )
        
        # Part 3: Locating the Target Shape
        target_cc_mask, highlighted_shape = part3_locating_target_shape(img_bgr, boundaries_no_blue, blue_mask)
        show_images_in_row(
            [target_cc_mask, highlighted_shape],
            ["Part 3.c: Target CC Mask", "Part 3.d: Highlighted Target Shape"]
        )

if __name__ == "__main__":
    main()
