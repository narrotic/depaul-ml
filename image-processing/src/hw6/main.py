import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

threshold_config = {
    "image1.jpg": {
        "gray": 110,
        "R": 140,
        "G": 100,
        "B": 90,
        "hue": (100, 130),
        "hist_ranges": [(0, 60), (61, 150), (151, 255)],
        "hue_ranges": [(0, 20), (50, 80), (100, 130)]
    },
    "image2.jpg": {
        "gray": 130,
        "R": 120,
        "G": 150,
        "B": 80,
        "hue": (30, 70),
        "hist_ranges": [(0, 70), (71, 180), (181, 255)],
        "hue_ranges": [(20, 40), (60, 90), (120, 160)]
    }
}

default_config = {
    "gray": 120,
    "R": 120,
    "G": 120,
    "B": 120,
    "hue": (0, 179),
    "hist_ranges": [(0, 85), (86, 170), (171, 255)],
    "hue_ranges": [(0, 60), (61, 120), (121, 179)]
}



def load_grayscale(path):
    """Load image and convert to grayscale (used in all parts)"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def show_images(images, titles, cols=4, cmap='gray'):
    """Utility function to display multiple images in a grid layout"""
    rows = int(np.ceil(len(images) / cols))
    # plt.figure(figsize=(8 * cols, 8 * rows))
    plt.figure(figsize=(14, 8))
    plt.tight_layout()
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_color(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def apply_mask_color(original, mask, bg_color=(0, 0, 255)):
    """
    Replace background pixels with fixed color.
    mask should be binary: 255 foreground, 0 background
    """
    result = original.copy()
    result[mask == 0] = bg_color
    return result

def manual_threshold(channel, thresh):
    _, mask = cv2.threshold(channel, thresh, 255, cv2.THRESH_BINARY)
    return mask

def otsu_threshold(channel):
    _, mask = cv2.threshold(channel, 0, 255, 
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def hue_interval_mask(hue_channel, lower, upper):
    """
    lower, upper in OpenCV hue range [0,179]
    Handles wrap-around
    """
    if lower <= upper:
        mask = cv2.inRange(hue_channel, lower, upper)
    else:
        mask1 = cv2.inRange(hue_channel, lower, 179)
        mask2 = cv2.inRange(hue_channel, 0, upper)
        mask = cv2.bitwise_or(mask1, mask2)
    return mask

def part1_task1_grayscale(color_img, gray_thresh):
    """
    Part 1 - Task 1
    1. Convert RGB image to grayscale using simple average
    2. Apply manual threshold
    3. Replace background pixels with fixed color
    """

    # Convert to grayscale using average of channels
    gray_avg = np.mean(color_img, axis=2).astype(np.uint8)

    # Apply manual threshold
    gray_mask = manual_threshold(gray_avg, gray_thresh)  # adjust per image

    # Apply mask to original color image
    gray_segmented = apply_mask_color(color_img, gray_mask)

    # Display results
    show_images(
        [color_img[:, :, ::-1], gray_avg, gray_mask, gray_segmented[:, :, ::-1]],
        ["Original", "Grayscale", "Gray Mask", "Gray Segmented"],
        cols=4
    )

def part1_task2_rgb(color_img, config):
    """
    Part 1 - Task 2
    Perform segmentation on R, G, B channels
    - Manual threshold
    - OTSU threshold
    """

    b, g, r = cv2.split(color_img)

    channels = {
        "R": r,
        "G": g,
        "B": b
    }

    for name, channel in channels.items():

        # Manual threshold
        manual_mask = manual_threshold(channel, config[name])
        manual_segmented = apply_mask_color(color_img, manual_mask)

        # OTSU threshold
        otsu_mask = otsu_threshold(channel)
        otsu_segmented = apply_mask_color(color_img, otsu_mask)

        show_images(
            [
                manual_mask,
                manual_segmented[:, :, ::-1],
                otsu_mask,
                otsu_segmented[:, :, ::-1]
            ],
            [
                f"{name} Manual Mask",
                f"{name} Manual Segmented",
                f"{name} OTSU Mask",
                f"{name} OTSU Segmented"
            ],
            cols=2
        )

def part1_task3_hsv(color_img, config):
    """
    Part 1 - Task 3
    1. Convert RGB to HSV
    2. Threshold Hue channel using interval
    3. Replace background pixels
    """

    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Example: blue object
    # Adjust range depending on object
    lower, upper = config["hue"]
    hue_mask = hue_interval_mask(h, lower, upper)

    hue_segmented = apply_mask_color(color_img, hue_mask)

    show_images(
        [
            color_img[:, :, ::-1],
            hue_mask,
            hue_segmented[:, :, ::-1]
        ],
        [
            "Original",
            "Hue Mask",
            "Hue Segmented"
        ],
        cols=3
    )

def part2_task1_histogram_gray(color_img, config):
    """
    Part 2 - Task 1
    Histogram based grayscale segmentation
    """

    gray = np.mean(color_img, axis=2).astype(np.uint8)

    # Show grayscale
    plt.figure()
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.show()

    # Plot histogram
    plt.figure()
    plt.hist(gray.ravel(), bins=256)
    plt.title("Grayscale Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()

    # Define intensity ranges
    ranges = config["hist_ranges"]

    masks = []
    for r_min, r_max in ranges:
        mask = cv2.inRange(gray, r_min, r_max)
        masks.append(mask)

        show_images(
            [mask],
            [f"Mask {r_min}-{r_max}"],
            cols=1
        )

    # Combine masks into colored segmented image
    segmented = np.zeros_like(color_img)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for mask, color in zip(masks, colors):
        segmented[mask == 255] = color

    show_images(
        [segmented[:, :, ::-1]],
        ["Histogram Segmented Image"],
        cols=1
    )

def part2_task2_histogram_hue(color_img, config):
    """
    Part 2 - Task 2
    Histogram based hue segmentation
    """

    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Show original
    show_images(
        [color_img[:, :, ::-1]],
        ["Original Color Image"],
        cols=1,
        cmap=None
    )

    # Plot hue histogram
    plt.figure()
    plt.hist(h.ravel(), bins=180)
    plt.title("Hue Histogram")
    plt.xlabel("Hue")
    plt.ylabel("Frequency")
    plt.show()

    # Define hue ranges (example ranges)
    hue_ranges = config["hue_ranges"]

    masks = []
    for lower, upper in hue_ranges:
        mask = hue_interval_mask(h, lower, upper)
        masks.append(mask)

        show_images(
            [mask],
            [f"Hue Mask {lower}-{upper}"],
            cols=1
        )

    segmented = np.zeros_like(color_img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for mask, color in zip(masks, colors):
        segmented[mask == 255] = color

    show_images(
        [segmented[:, :, ::-1]],
        ["Hue Histogram Segmented Image"],
        cols=1
    )

def main():
    base = Path(__file__).resolve().parents[2] / "data" / "hw6"
    # images = list(base.glob("*.jpg")) + list(base.glob("*.jpeg")) + list(base.glob("*.png"))
    images = [
    p for p in base.glob("*.*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    and not p.name.startswith("bg_")
    ]


    if not images:
        print(f"No images found in {base}")
        return

    print(f"Found {len(images)} image(s) to process")

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        color_img = load_color(img_path)

        print("Part 1 - Task 1")
        config = threshold_config.get(img_path.name, default_config)
        part1_task1_grayscale(color_img, config["gray"])

        print("Part 1 - Task 2")
        part1_task2_rgb(color_img, config)

        print("Part 1 - Task 3")
        part1_task3_hsv(color_img, config)

        print("Part 2 - Task 1")
        part2_task1_histogram_gray(color_img, config)

        print("Part 2 - Task 2")
        part2_task2_histogram_hue(color_img, config)

if __name__ == "__main__":
    main()