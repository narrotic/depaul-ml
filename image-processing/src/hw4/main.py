import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

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

def spatial_filtering_demo(gray):
    # a) grayscale already given
    # b) mean threshold
    T = gray.mean()
    binary = (gray > T).astype(np.uint8) * 255

    # c) structuring element
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # d) two erosions
    erosion = cv2.erode(binary, se, iterations=2)

    # e) two dilations
    dilation = cv2.dilate(binary, se, iterations=2)

    # f) difference
    diff = cv2.absdiff(dilation, erosion)

    # display
    show_images(
        [gray, binary, erosion, dilation, diff],
        [
            "Grayscale",
            "Binary (Mean Threshold)",
            "Binary after 2 Erosions",
            "Binary after 2 Dilations",
            "Difference (Dilation − Erosion)"
        ]
    )

def noise_reduction_demo(gray):
    T = gray.mean()
    binary = (gray > T).astype(np.uint8) * 255

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se, iterations=2)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se, iterations=2)

    diff = cv2.absdiff(closing, opening)

    show_images(
        [gray, binary, opening, closing, diff],
        [
            "Grayscale",
            "Binary (Mean Threshold)",
            "Binary after 2 Openings",
            "Binary after 2 Closings",
            "Difference (Closing − Opening)"
        ]
    )

def custom_sobel_filter_demo(gray, sigma=1.0):
    T = gray.mean()
    binary = (gray > T).astype(np.uint8) * 255

    # c) disk 5x5
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # d) boundary extraction
    eroded = cv2.erode(binary, se)
    boundary = cv2.subtract(binary, eroded)

    # e) Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel_x = np.abs(sobel_x)
    sobel_y = np.abs(sobel_y)

    tx = sobel_x.mean()
    ty = sobel_y.mean()

    _, sx = cv2.threshold(sobel_x, tx, 255, cv2.THRESH_BINARY)
    _, sy = cv2.threshold(sobel_y, ty, 255, cv2.THRESH_BINARY)

    sx = sx.astype(np.uint8)
    sy = sy.astype(np.uint8)

    combined = cv2.bitwise_or(sx, sy)

    show_images(
        [gray, binary, eroded, boundary, sx, sy, combined],
        [
            "Grayscale",
            "Binary",
            "Eroded Image",
            "Morphological Boundary",
            "Sobel Horizontal",
            "Sobel Vertical",
            "Sobel Combined"
        ],
        cols=4
    )


def photomontage_demo(fg_color, bg_color):
    # b) Grayscale copy
    fg_gray = cv2.cvtColor(fg_color, cv2.COLOR_BGR2GRAY)

    # c) Segmentation (OTSU + direction check)
    _, mask = cv2.threshold(
        fg_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure foreground is white
    if fg_gray.mean() > 127:
        mask = cv2.bitwise_not(mask)

    # d) Refine mask
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=2)
    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, se, iterations=1)

    # e) Initial photomontage
    bg_resized = cv2.resize(bg_color, (fg_color.shape[1], fg_color.shape[0]))
    montage = bg_resized.copy()
    montage[mask_refined == 255] = fg_color[mask_refined == 255]

    # f) Segmentation borders
    se_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(mask_refined, se_border)
    eroded = cv2.erode(mask_refined, se_border)
    border = cv2.subtract(dilated, eroded)

    # g) Smooth seams only
    blurred = cv2.GaussianBlur(montage, (15, 15), 0)
    final = montage.copy()
    final[border == 255] = blurred[border == 255]

    show_images(
        [
            cv2.cvtColor(fg_color, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(bg_color, cv2.COLOR_BGR2RGB), # Added BG
            fg_gray,                                    # Added Grayscale
            mask,
            mask_refined,
            cv2.cvtColor(montage, cv2.COLOR_BGR2RGB),
            border,
            cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        ],
        [
            "Input Foreground", "Input Background", "Grayscale FG",
            "Initial Segmentation", "Refined Segmentation", "Initial Photomontage",
            "Segmentation Border", "Full Blurred Montage", "Final Refined Montage"
        ],
        cols=3,
        cmap='gray'
    )

def photomontage_wrapper(img_path):
    fg_color = load_color(img_path)

    # Try to find background with the same name, any extension
    base_name = img_path.stem  # e.g., 'white-mug'
    bg_candidates = list(img_path.parent.glob(f"bg_{base_name}.*"))

    if not bg_candidates:
        print(f"Background not found for {img_path.name}, skipping Part 4")
        return

    bg_color = load_color(bg_candidates[0])  # pick first match

    photomontage_demo(fg_color, bg_color)


def load_color(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def main():
    base = Path(__file__).resolve().parents[2] / "data" / "hw4"
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
        img = load_grayscale(img_path)

        print("Part 1: Erosion & Dilation")
        spatial_filtering_demo(img)

        print("Part 2: Opening & Closing")
        noise_reduction_demo(img)

        print("Part 3: Boundary Extraction")
        custom_sobel_filter_demo(img)

        print("Part 4: Photomontage")
        photomontage_wrapper(img_path)

if __name__ == "__main__":
    main()