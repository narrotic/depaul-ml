import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_grayscale(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Could not load image: {path}")
    return img

def show_images(images, titles, cols=4, cmap='gray'):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(16, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        # Handle RGB vs Gray for matplotlib
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap=cmap)
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Part 1: Edge Detectors ---

def apply_roberts(img):
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(img, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(img, cv2.CV_32F, kernel_y)
    return cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))

def apply_prewitt(img):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(img, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(img, cv2.CV_32F, kernel_y)
    return cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))

def apply_sobel(img, return_components=False):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))

    if return_components:
        return gx, gy, mag
    return mag

def apply_canny(img, t1=50, t2=150):
    # Gaussian blur is often recommended before Canny to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.Canny(blurred, t1, t2)

# --- Logic for Assignment Parts ---

def part1_edges(img_gray):
    rob = apply_roberts(img_gray)
    pre = apply_prewitt(img_gray)
    sob = apply_sobel(img_gray)
    can = apply_canny(img_gray) # Adjust t1, t2 per image as needed
    
    titles = ["Original Gray", "Roberts", "Prewitt", "Sobel", "Canny"]
    show_images([img_gray, rob, pre, sob, can], titles)
    return can # Canny is 'best' for the next step

def part2_color_edges(img_bgr, detector_func):
    # RGB Channels
    b, g, r = cv2.split(img_bgr)
    edge_r = detector_func(r)
    edge_g = detector_func(g)
    edge_b = detector_func(b)
    
    # HSV Channels
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    edge_h = detector_func(h)
    edge_v = detector_func(v)
    
    titles = ["Original", "Edges (Red)", "Edges (Green)", "Edges (Blue)", "Edges (Hue)", "Edges (Value/Int)"]
    show_images([img_bgr, edge_r, edge_g, edge_b, edge_h, edge_v], titles)

def part3_highlighting(img_bgr, edge_map):
    # c) Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thick_edges = cv2.dilate(edge_map, kernel)
    
    # d) Invert
    _, regions = cv2.threshold(cv2.bitwise_not(thick_edges), 127, 255, cv2.THRESH_BINARY)
    
    # e) Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(regions)
    
    # f) Find largest region (ignoring the background index 0 if it's just the border)
    # Usually, the largest area in stats[1:] is our target
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    s = stats[largest_label]
    
    print(f"Largest Region Stats: Area={s[cv2.CC_STAT_AREA]}, "
          f"Bounds: ({s[cv2.CC_STAT_LEFT]}, {s[cv2.CC_STAT_TOP]}) to "
          f"({s[cv2.CC_STAT_LEFT]+s[cv2.CC_STAT_WIDTH]}, {s[cv2.CC_STAT_TOP]+s[cv2.CC_STAT_HEIGHT]})")
    
    # g) Largest region mask
    mask_largest = (labels == largest_label).astype(np.uint8) * 255
    
    # h) Inverted mask (background)
    mask_bg = cv2.bitwise_not(mask_largest)
    
    # i) j) Color processing
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Set saturation to 0 where mask_bg is white (255)
    s[mask_bg == 255] = 0
    
    final_hsv = cv2.merge([h, s, v])
    final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    titles = ["Original", "Initial Edges", "Thickened", "Regions Mask", 
              "Largest Mask", "Background Mask", "Final Highlight"]
    show_images([img_bgr, edge_map, thick_edges, regions, mask_largest, mask_bg, final_rgb], titles, cols=4)

def main():
    base = Path("data/hw5") 
    image_paths = list(base.glob("*.jpg")) + list(base.glob("*.jpeg"))
    
    for path in image_paths:
        print(f"\n--- Processing {path.name} ---")
        img_bgr = cv2.imread(str(path))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        print("Part 1: Running Edge Detectors...")
        best_edge = part1_edges(img_gray)
        
        print("Part 2: Color Space Edge Detection...")
        part2_color_edges(img_bgr, apply_canny)
        
        print("Part 3: Region Highlighting...")
        part3_highlighting(img_bgr, best_edge)

if __name__ == "__main__":
    main()