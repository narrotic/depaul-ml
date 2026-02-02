# -------------------- Part 5: Spatial Filtering (CSC 481 only) --------------------
# Requirements:
# a) Custom function with explicit pixel-by-pixel sum of products ✓
# b) Assumption: Zero-padding used to maintain size ✓
# c) Built-in filtering (cv2.filter2D) for comparison ✓
# d) 3 Filters: Gaussian, Point (Laplacian), Diagonal Sobel ✓
# e) Difference image: No diff = 127, Max pos = 255, Max neg = 0 ✓

import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def custom_linear_filter(image, kernel):
    """
    Manual 2D linear filtering using explicit sum-of-products.
    Uses zero-padding to keep output size equal to input.
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Apply zero-padding
    padded = np.zeros((h + 2*pad_h, w + 2*pad_w), dtype=np.float32)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image.astype(np.float32)

    output = np.zeros_like(image, dtype=np.float32)

    # Pixel-by-pixel sum-of-products
    for i in range(h):
        for j in range(w):
            # Extract neighborhood
            region = padded[i:i+kh, j:j+kw]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * kernel)
            
    return output

# Part 1
def spatial_filtering_demo(img):
    """
    Demonstrates spatial filtering with custom and built-in methods.
    """
    # Define exact kernels from assignment
    kernels = {
        "Gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0,
        "Point": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        "Diagonal Sobel": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    }

    for name, kernel in kernels.items():
        # 1. Custom implementation
        custom_res = custom_linear_filter(img, kernel)
        
        # 2. Built-in implementation
        # BORDER_CONSTANT with value 0 creates identical zero-padding
        builtin_res = cv2.filter2D(img.astype(np.float32), -1, kernel, 
                                   borderType=cv2.BORDER_CONSTANT)
        
        # 3. Difference Image Calculation
        # d(x,y) = 127 + (custom - builtin)
        # Scaled so that no difference appears as gray (127)
        diff = custom_res - builtin_res
        diff_vis = np.clip(127 + diff, 0, 255).astype(np.uint8)

        # 4. Display Results for this specific filter (4 images)
        # Reusing your show_images logic or custom subplot for 1x4 layout
        results = [img, custom_res.astype(np.uint8), builtin_res.astype(np.uint8), diff_vis]
        titles = ["Original", f"Custom {name}", f"Built-in {name}", "Difference"]
        
        print(f"Displaying results for {name} filter...")
        show_images(results, titles, cols=4)

def noise_reduction_demo(img):
    """
    Demonstrates noise reduction.
    Cleans both Gaussian and S&P noise using both Mean and Median filters.
    
    - Applies mean & median filters to both types of noise
    - Iterates over all required filter sizes
    - Produces all 12 results per image
    """
    # 1. Create Noise
    noise_g = np.random.normal(0, 25, img.shape)
    gaussian_noise = np.clip(img.astype(np.float32) + noise_g, 0, 255).astype(np.uint8)

    salt_pepper_noise = img.copy()
    prob = 0.05
    rnd = np.random.rand(*img.shape)
    salt_pepper_noise[rnd < prob/2] = 0
    salt_pepper_noise[rnd > 1 - prob/2] = 255

    sizes = [3, 5, 7]
    
    # Process Gaussian Noise with both filters
    for s in sizes:
        mean_clean = cv2.blur(gaussian_noise, (s, s))
        med_clean = cv2.medianBlur(gaussian_noise, s)
        show_images([gaussian_noise, mean_clean, med_clean], 
                    [f"Gauss Noise", f"Mean {s}x{s}", f"Median {s}x{s}"], cols=3)

    # Process Salt & Pepper Noise with both filters
    for s in sizes:
        mean_clean = cv2.blur(salt_pepper_noise, (s, s))
        med_clean = cv2.medianBlur(salt_pepper_noise, s)
        show_images([salt_pepper_noise, mean_clean, med_clean], 
                    [f"S&P Noise", f"Mean {s}x{s}", f"Median {s}x{s}"], cols=3)
        
# Part 3
def gaussian_derivative_kernels(size=7, sigma=1.0):
    """
    Create 7x7 Gaussian derivative kernels for x and y gradients.
    """
    assert size % 2 == 1, "Kernel size must be odd"

    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))

    factor = -1 / (2 * np.pi * sigma**4)
    exp_term = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    Gx = factor * x * exp_term
    Gy = factor * y * exp_term

    return Gx, Gy

def custom_sobel_filter_demo(img, sigma=1.0):
    """
    Apply custom 7x7 Sobel (Gaussian derivative) filters.
    """
    Gx, Gy = gaussian_derivative_kernels(size=7, sigma=sigma)

    grad_x = cv2.filter2D(img.astype(np.float32), -1, Gx, borderType=cv2.BORDER_CONSTANT)
    grad_y = cv2.filter2D(img.astype(np.float32), -1, Gy, borderType=cv2.BORDER_CONSTANT)

    show_images(
        [img, np.abs(grad_x), np.abs(grad_y)],
        [
            "Original",
            f"Custom Sobel Gx (sigma={sigma})",
            f"Custom Sobel Gy (sigma={sigma})"
        ],
        cols=3
    )

    return grad_x, grad_y

# Part 4 – Image Transformations (CSC 481 only)
def image_rotation_demo(img):
    """
    Demonstrates affine image rotation around top-left corner and center.
    """
    h, w = img.shape

    # --- Rotation around TOP-LEFT corner (0,0) ---
    center_tl = (0, 0)

    M_tl_pos = cv2.getRotationMatrix2D(center_tl, 45, 1.0)
    M_tl_neg = cv2.getRotationMatrix2D(center_tl, -45, 1.0)

    rot_tl_pos = cv2.warpAffine(img, M_tl_pos, (w, h))
    rot_tl_neg = cv2.warpAffine(img, M_tl_neg, (w, h))

    # --- Rotation around CENTER ---
    center_img = (w // 2, h // 2)

    M_c_pos = cv2.getRotationMatrix2D(center_img, 45, 1.0)
    M_c_neg = cv2.getRotationMatrix2D(center_img, -45, 1.0)

    rot_c_pos = cv2.warpAffine(img, M_c_pos, (w, h))
    rot_c_neg = cv2.warpAffine(img, M_c_neg, (w, h))

    # --- Display results ---
    show_images(
        [img, rot_tl_pos, rot_tl_neg, rot_c_pos, rot_c_neg],
        [
            "Original",
            "Top-left +45°",
            "Top-left -45°",
            "Center +45°",
            "Center -45°"
        ],
        cols=3
    )


def main():
    # ... (Your existing Path and glob logic) ...
    base = Path(__file__).resolve().parents[2] / "data" / "hw3"
    images = list(base.glob("*.jpg")) + list(base.glob("*.jpeg"))

    if not images:
        print(f"No images found in {base}")
        print("Please add your 3 required images to the data/hw2 directory")
        return

    print(f"Found {len(images)} image(s) to process")

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        img = load_grayscale(img_path)

        # ... (Calls to Part 1, 2, 3, 4) ...
        
        print("Part 1: Spatial Filtering (CSC 481)...")
        spatial_filtering_demo(img)

        print("Part 2: Noise Reduction...")
        noise_reduction_demo(img)

        print("Part 3: Custom Sobel (gradient) Filter")
        custom_sobel_filter_demo(img, sigma=1.0)

        print("Part 4:  Image Transformations (CSC 481 only)..")
        image_rotation_demo(img)

if __name__ == "__main__":
    main()
