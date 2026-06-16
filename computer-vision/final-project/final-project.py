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
import time


def load_color(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def show_images(images, titles, cols=4, cmap="gray"):
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
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_uniform(path, img, target_width=1280, jpg_quality=92):
    """
    Writes an image to disk at a canonical pixel width while preserving
    aspect ratio. Keeping every saved figure at the same width guarantees a
    consistent object-to-frame ratio across the dataset, so side-by-side
    inspection of detection outputs is not skewed by per-image resolution
    differences. Single-channel masks are saved as-is; colour images use the
    requested JPEG quality. Returns the path actually written.
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    if w != target_width and w > 0:
        scale = target_width / float(w)
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        img = cv2.resize(img, (target_width, new_h), interpolation=interp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"} and img.ndim == 3:
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(str(path), img)
    return path


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
        img_bgr[mask_indices], alpha, green_overlay[mask_indices], 1 - alpha, 0
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
    mask_closed = cv2.morphologyEx(
        mask_opened, cv2.MORPH_CLOSE, kernel_large, iterations=2
    )

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
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
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
            if i in matched_gt:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - len(matched_gt)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1_score


def compute_bbox_iou_mean(pred_boxes, gt_boxes):
    """
    For every ground-truth box, take the IoU of the best matching prediction
    and return the mean. Captures localisation quality independently of the
    F1 0.5 threshold (e.g. boxes that are 'mostly right' but not quite at IoU 0.5).
    """
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0
    ious = [max(compute_iou(p, g) for p in pred_boxes) for g in gt_boxes]
    return float(np.mean(ious))


def parse_ground_truth(json_path, image_name):
    """
    Placeholder utility to parse ground truth bounding boxes for an image.
    Expects format: {"image_name.jpg": [[x, y, w, h], ...]}
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get(image_name, [])
    except FileNotFoundError:
        print(f"Ground truth file not found at {json_path}. Returning empty list.")
        return []


def compute_pr_curve_voc(detections_by_image, gt_by_image, iou_threshold=0.5):
    """
    Pascal-VOC-style precision/recall curve and Average Precision for a
    single class (here: 'person'). All detections across the dataset are
    pooled and sorted by descending score; for each detection in that order
    we mark it as TP (matches an unmatched GT at IoU >= threshold) or FP,
    accumulate (TP, FP), and produce one (precision, recall) pair per rank.

    AP is the area under the all-points interpolated PR curve
    (Pascal VOC 2010+ convention): at every recall r, precision is the
    maximum precision attained at any recall >= r.

    detections_by_image : dict[str, list[(box, score)]]
    gt_by_image         : dict[str, list[box]]
    Returns: precision (np.ndarray), recall (np.ndarray), ap (float).
    """
    all_dets = []
    for name, dets in detections_by_image.items():
        for box, score in dets:
            all_dets.append((float(score), name, box))
    all_dets.sort(key=lambda x: -x[0])

    n_gt = sum(len(v) for v in gt_by_image.values())
    if n_gt == 0 or not all_dets:
        return np.array([1.0]), np.array([0.0]), 0.0

    matched = {k: set() for k in gt_by_image}
    tp = np.zeros(len(all_dets), dtype=np.float64)
    fp = np.zeros(len(all_dets), dtype=np.float64)

    for k, (_score, name, box) in enumerate(all_dets):
        gts = gt_by_image.get(name, [])
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gts):
            if i in matched[name]:
                continue
            iou = compute_iou(box, g)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold and best_idx >= 0:
            tp[k] = 1.0
            matched[name].add(best_idx)
        else:
            fp[k] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / float(n_gt)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # All-points interpolation: monotone non-increasing precision envelope
    p_interp = np.maximum.accumulate(precision[::-1])[::-1]
    # Prepend (r=0, p=p_interp[0]) so the AUC starts at recall 0
    r_ext = np.concatenate(([0.0], recall))
    p_ext = np.concatenate(([p_interp[0]], p_interp))
    # NumPy 2.x renamed trapz -> trapezoid; fall back for older runtimes.
    _trapz = getattr(np, "trapezoid", np.trapz)
    ap = float(_trapz(p_ext, r_ext))
    return precision, recall, ap


def plot_pr_curves(
    curves, save_path, title="Precision-Recall Curves (person, IoU = 0.5)"
):
    """
    Saves a publication-style PR plot containing one or more curves. Each
    entry in `curves` is a dict with keys {name, precision, recall, ap,
    color, score_kind}; `score_kind` is a short string used in the legend
    to disclose the ranking signal (e.g. "model conf.", "mean prob.",
    "fill ratio"). The iso-F1 = 0.5 contour is drawn as a grey dashed
    reference so the audience can see how far each curve sits above the
    weak-detector band.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    r_iso = np.linspace(0.34, 1.0, 200)
    p_iso = (0.5 * r_iso) / np.clip(2 * r_iso - 0.5, 1e-6, None)
    ax.plot(
        r_iso,
        np.clip(p_iso, 0, 1),
        linestyle="--",
        color="#bdbdbd",
        linewidth=1.0,
        label="iso-F1 = 0.5",
    )
    for c in curves:
        label = f"{c['name']}   AP = {c['ap']:.3f}  ({c['score_kind']})"
        ax.plot(
            c["recall"], c["precision"], color=c["color"], linewidth=2.2, label=label
        )
        ax.fill_between(c["recall"], 0, c["precision"], color=c["color"], alpha=0.08)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_title(title, fontsize=13)
    ax.grid(which="major", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_pr_curve(precision, recall, ap, save_path, title="Mask R-CNN (person)"):
    """Single-curve wrapper retained for backwards compatibility."""
    return plot_pr_curves(
        [
            {
                "name": title,
                "precision": precision,
                "recall": recall,
                "ap": ap,
                "color": "#1f77b4",
                "score_kind": "model conf.",
            }
        ],
        save_path,
    )


def harvest_maskrcnn_detections(img_bgr, low_score=0.05):
    """
    Runs Mask R-CNN once at a very low confidence so we keep the entire
    score distribution. Returned as list[(box, score)] for the 'person'
    class only; used to build the dataset-level PR curve and AP.
    """
    _, boxes, scores = segment_players_maskrcnn(img_bgr, score_threshold=low_score)
    return list(zip(boxes, scores))


def harvest_deeplab_detections(img_bgr, refine=True):
    """
    Produces (box, score) pairs for the semantic-segmentation pipeline. Since
    DeepLabV3 has no per-instance confidence, we use the mean person-class
    softmax probability inside each connected-component blob as a proxy
    score. Blobs with higher mean person-probability are treated as more
    confident detections when sweeping thresholds for the PR curve.
    """
    raw_mask, person_prob = segment_players_dl_with_probs(img_bgr)
    mask = refine_dl_mask(raw_mask) if refine else raw_mask
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if w <= 0 or h <= 0 or area < 80:
            continue
        comp_mask = labels[y : y + h, x : x + w] == i
        if not np.any(comp_mask):
            continue
        score = float(person_prob[y : y + h, x : x + w][comp_mask].mean())
        out.append(((x, y, w, h), score))
    return out


def harvest_classical_detections(img_bgr, method="hsv"):
    """
    Produces (box, score) pairs for the classical pipeline. Classical CV is
    deterministic and emits no confidence, so we rank blobs by their
    bounding-box fill ratio (foreground pixels inside the bbox / bbox
    area). A tight, dense blob fills its box and is treated as a more
    confident player detection; sparse or noisy blobs score lower. This
    proxy is monotonic in compactness, which is the property the classical
    morphology pipeline is actually optimising for.
    """
    if method == "kmeans":
        fg = extract_features_kmeans(img_bgr)
    else:
        _, hsv = preprocess_image(img_bgr)
        fg, _ = extract_features(hsv)
    boxes = detect_players(fg)
    fg_bin = (fg > 0).astype(np.uint8)
    out = []
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        roi = fg_bin[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        fill = float(roi.sum()) / float(roi.size)
        out.append(((x, y, w, h), fill))
    return out


def segment_players_dl_with_probs(img_bgr):
    """
    Same forward pass as `segment_players_dl` but additionally returns the
    softmax probability map for the 'person' class. The argmax mask is the
    standard binary segmentation output; the probability map is used by
    `harvest_deeplab_detections` to rank connected components for the
    cross-pipeline PR plot.
    """
    import torch

    model, device = _get_dl_model()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    batch = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)["out"]
        probs = torch.softmax(logits, dim=1)[0, _PERSON_CLASS_INDEX].cpu().numpy()
        preds = logits.argmax(dim=1)[0].cpu().numpy()
    mask = np.where(preds == _PERSON_CLASS_INDEX, 255, 0).astype(np.uint8)
    return mask, probs.astype(np.float32)


# --- Module 5: Deep Learning Segmentation (Transfer Learning) ---
# DeepLabV3-ResNet50 pretrained on COCO with VOC label semantics.
# In the VOC label set used by torchvision, "person" is class 15.
_DL_MODEL = None
_DL_DEVICE = None
_PERSON_CLASS_INDEX = 15


def _get_dl_model():
    """Lazily loads the DeepLabV3-ResNet50 model (singleton)."""
    global _DL_MODEL, _DL_DEVICE
    if _DL_MODEL is not None:
        return _DL_MODEL, _DL_DEVICE

    import torch
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        DeepLabV3_ResNet50_Weights,
    )

    if torch.cuda.is_available():
        _DL_DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _DL_DEVICE = torch.device("mps")
    else:
        _DL_DEVICE = torch.device("cpu")

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    _DL_MODEL = deeplabv3_resnet50(weights=weights).eval().to(_DL_DEVICE)
    print(f"Loaded DeepLabV3-ResNet50 (pretrained, COCO/VOC) on device: {_DL_DEVICE}")
    return _DL_MODEL, _DL_DEVICE


def segment_players_dl(img_bgr):
    """
    Runs DeepLabV3-ResNet50 on the image and returns a binary foreground mask
    containing only the 'person' class. Output is uint8 in {0, 255}, same H,W as input.
    """
    import torch

    model, device = _get_dl_model()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    batch = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(batch)["out"]  # (1, num_classes, H, W)

    preds = out.argmax(dim=1)[0].cpu().numpy()
    dl_mask = np.where(preds == _PERSON_CLASS_INDEX, 255, 0).astype(np.uint8)
    return dl_mask


# --- Mask R-CNN (instance segmentation, transfer learning on COCO) ---
_MRCNN_MODEL = None
_MRCNN_DEVICE = None
_COCO_PERSON_CLASS_INDEX = 1  # torchvision detection: 'person' has COCO label 1


def _get_maskrcnn_model():
    """Lazily loads Mask R-CNN ResNet50-FPN with default COCO weights (singleton)."""
    global _MRCNN_MODEL, _MRCNN_DEVICE
    if _MRCNN_MODEL is not None:
        return _MRCNN_MODEL, _MRCNN_DEVICE

    import torch
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn,
        MaskRCNN_ResNet50_FPN_Weights,
    )

    if torch.cuda.is_available():
        _MRCNN_DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _MRCNN_DEVICE = torch.device("mps")
    else:
        _MRCNN_DEVICE = torch.device("cpu")

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    _MRCNN_MODEL = maskrcnn_resnet50_fpn(weights=weights).eval().to(_MRCNN_DEVICE)
    print(f"Loaded MaskRCNN-ResNet50-FPN (pretrained, COCO) on device: {_MRCNN_DEVICE}")
    return _MRCNN_MODEL, _MRCNN_DEVICE


def segment_players_maskrcnn(img_bgr, score_threshold=0.6, mask_threshold=0.5):
    """
    Instance segmentation with Mask R-CNN, restricted to the COCO 'person' class.
    Returns (mask, boxes, scores):
      - mask  : union of the kept instance masks, uint8 in {0, 255}, same H,W as input
      - boxes : list of (x, y, w, h) ints
      - scores: list of confidences aligned with boxes
    """
    import torch

    model, device = _get_maskrcnn_model()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        outputs = model([tensor.to(device)])

    out = outputs[0]
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    boxes_xyxy = out["boxes"].cpu().numpy()
    masks = out["masks"].cpu().numpy()  # (N, 1, H, W) probabilities

    H, W = img_bgr.shape[:2]
    fg_mask = np.zeros((H, W), dtype=np.uint8)
    kept_boxes, kept_scores = [], []
    for i, (lab, score) in enumerate(zip(labels, scores)):
        if lab != _COCO_PERSON_CLASS_INDEX or score < score_threshold:
            continue
        m = (masks[i, 0] > mask_threshold).astype(np.uint8) * 255
        fg_mask = np.maximum(fg_mask, m)
        x1, y1, x2, y2 = boxes_xyxy[i]
        x = int(round(x1))
        y = int(round(y1))
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))
        if w > 0 and h > 0:
            kept_boxes.append((x, y, w, h))
            kept_scores.append(float(score))
    return fg_mask, kept_boxes, kept_scores


def refine_dl_mask(mask, close_kernel_size=7, min_blob_area=80, blur_ksize=0):
    """
    Post-processes a binary DL mask: morphological closing -> small-blob removal
    -> optional Gaussian smoothing. Improves mask compactness for downstream IoU.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    if min_blob_area > 0:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(closed)
        keep = np.zeros_like(closed)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                keep[lbl == i] = 255
        closed = keep

    if blur_ksize > 0:
        b = blur_ksize | 1
        closed = cv2.GaussianBlur(closed, (b, b), 0)
        closed = (closed > 127).astype(np.uint8) * 255

    return closed


def mask_to_bboxes(mask, min_area=100, max_area=20000):
    """
    Generic binary-mask -> bounding-box converter using connected components.
    Slightly more permissive than detect_players because DL masks are cleaner
    and tend to merge fewer non-player blobs.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect_ratio = float(w) / max(h, 1)
        if min_area < area < max_area and 0.15 < aspect_ratio < 3.0:
            boxes.append((x, y, w, h))
    return boxes


def compute_mask_iou(mask1, mask2):
    """
    Intersection over Union between two binary masks of identical shape.

    Caveat for this project: the ground-truth mask is a union of GT bounding
    rectangles (see `gt_boxes_to_mask`), not pixel-accurate silhouettes.
    Instance segmenters (Mask R-CNN) predict tight player silhouettes that
    occupy roughly 50-60% of each GT rectangle, so their Mask IoU here is
    effectively *lower-bounded by the rectangle-vs-silhouette mismatch* and
    cannot reach 1.0 even on a perfect prediction. Use this metric to
    compare detectors against each other, not against an absolute target.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"Mask shapes differ: {mask1.shape} vs {mask2.shape}")
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def gt_boxes_to_mask(gt_boxes, height, width):
    """Builds a pseudo ground-truth foreground mask from GT bounding boxes."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in gt_boxes:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(width, int(x + w))
        y1 = min(height, int(y + h))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def print_pipeline_insights(metrics_classical, metrics_dl, timings, mask_ious):
    """Compact qualitative analysis derived from the quantitative results."""
    _, _, f_c = metrics_classical
    _, _, f_d = metrics_dl
    t_c, t_d = timings
    iou_c, iou_d = mask_ious

    print("\n--- Analysis ---")
    if f_d > f_c + 0.05:
        print(
            "  DL outperforms classical CV: likely benefits from learned features that"
        )
        print("  handle occlusion, varied jersey colors, shadows, and crowd density.")
    elif f_c > f_d + 0.05:
        print(
            "  Classical CV outperforms DL on this image: field is well separated and"
        )
        print("  players are distinct, so HSV thresholding is sufficient.")
    else:
        print("  Both pipelines perform comparably on this image.")

    if iou_d > iou_c + 0.05:
        print(
            f"  DL mask aligns more tightly with GT regions (mask IoU {iou_d:.3f} vs {iou_c:.3f})."
        )
    elif iou_c > iou_d + 0.05:
        print(
            f"  Classical mask aligns more tightly with GT regions (mask IoU {iou_c:.3f} vs {iou_d:.3f})."
        )

    if t_c < t_d:
        speedup = t_d / max(t_c, 1e-6)
        print(
            f"  Classical pipeline is {speedup:.1f}x faster ({t_c*1000:.1f} ms vs {t_d*1000:.1f} ms)."
        )
    else:
        speedup = t_c / max(t_d, 1e-6)
        print(
            f"  DL pipeline is {speedup:.1f}x faster ({t_d*1000:.1f} ms vs {t_c*1000:.1f} ms)."
        )


# --- Module 6: Robustness perturbations ---
def apply_perturbations(img_bgr, kind="bright", strength=1.0, rng=None):
    """
    Synthetic perturbations to probe robustness of each pipeline.
    kind in {"bright", "dark", "noise", "blur"}; strength scales the effect.
    """
    if kind == "bright":
        return cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=int(50 * strength))
    if kind == "dark":
        return cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=int(-50 * strength))
    if kind == "noise":
        rng = rng if rng is not None else np.random.default_rng(0)
        sigma = 15.0 * strength
        noise = rng.normal(0, sigma, img_bgr.shape).astype(np.float32)
        return np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "blur":
        k = int(2 * round(3 * strength) + 1)
        return cv2.GaussianBlur(img_bgr, (k, k), 0)
    raise ValueError(f"Unknown perturbation kind: {kind}")


# --- Module 7: Centroid tracking ---
def track_players(prev_tracks, current_boxes, max_dist=60, next_id=0):
    """
    Greedy centroid tracker.
    prev_tracks  : list of (track_id, (x, y, w, h)) from the previous frame.
    current_boxes: list of (x, y, w, h) for this frame.
    Returns (new_tracks, next_id) where new_tracks aligns IDs to current_boxes.
    Boxes that do not match any previous track within max_dist get a fresh ID.
    """

    def _centroid(b):
        x, y, w, h = b
        return (x + w / 2.0, y + h / 2.0)

    if not prev_tracks:
        new_tracks = [(next_id + i, b) for i, b in enumerate(current_boxes)]
        return new_tracks, next_id + len(current_boxes)

    prev = [(tid, _centroid(b)) for tid, b in prev_tracks]
    new_tracks = []
    used = set()
    for b in current_boxes:
        cx, cy = _centroid(b)
        best_tid, best_d = None, max_dist + 1
        for tid, (px, py) in prev:
            if tid in used:
                continue
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if d < best_d:
                best_d, best_tid = d, tid
        if best_tid is not None and best_d <= max_dist:
            used.add(best_tid)
            new_tracks.append((best_tid, b))
        else:
            new_tracks.append((next_id, b))
            next_id += 1
    return new_tracks, next_id


def _hungarian_min_cost(cost):
    """
    O(n^3) Jonker-Volgenant assignment on a possibly-rectangular cost matrix.
    The matrix is padded to square with a large dummy cost so that real
    rows/columns are matched first. Returns an array `assignment` of length M
    (number of rows) where `assignment[i]` is the column matched to row i,
    or -1 if row i ended up matched to a dummy column (i.e. unmatched).

    No external dependency on scipy; standard potential-based shortest-
    augmenting-path Hungarian algorithm.
    """
    C = np.asarray(cost, dtype=np.float64)
    M, N = C.shape
    n = max(M, N)
    BIG = 1e9
    sq = np.full((n, n), BIG, dtype=np.float64)
    sq[:M, :N] = C

    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = sq[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = -np.ones(M, dtype=int)
    for j in range(1, n + 1):
        i = p[j] - 1
        if 0 <= i < M and 0 <= (j - 1) < N:
            assignment[i] = j - 1
    return assignment


def track_players_hungarian(
    prev_tracks, current_boxes, max_dist=60, iou_weight=0.5, next_id=0
):
    """
    Global-optimal centroid+IoU tracker. The cost of assigning previous
    track i to current detection j is

        c_ij = w * (1 - IoU(b_i, b_j)) + (1 - w) * (d_ij / max_dist)

    with `w = iou_weight`. Pairs whose centroid distance exceeds `max_dist`
    are gated out (set to a large infeasible cost) and never matched. The
    Hungarian algorithm finds the globally minimum-cost matching in one shot,
    fixing the swap/identity issues that the greedy nearest-neighbour
    tracker exhibits in crowded regions.
    """
    if not current_boxes:
        return [], next_id
    if not prev_tracks:
        new_tracks = [(next_id + i, b) for i, b in enumerate(current_boxes)]
        return new_tracks, next_id + len(current_boxes)

    def _centroid(b):
        x, y, w, h = b
        return (x + w / 2.0, y + h / 2.0)

    M = len(prev_tracks)
    N = len(current_boxes)
    INF_COST = 1e6
    cost = np.full((M, N), INF_COST, dtype=np.float64)
    for i, (_, pb) in enumerate(prev_tracks):
        pcx, pcy = _centroid(pb)
        for j, cb in enumerate(current_boxes):
            ccx, ccy = _centroid(cb)
            d = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5
            if d > max_dist:
                continue
            iou = compute_iou(pb, cb)
            cost[i, j] = iou_weight * (1.0 - iou) + (1.0 - iou_weight) * (d / max_dist)

    assignment = _hungarian_min_cost(cost)

    new_tracks = [None] * N
    for i, j in enumerate(assignment):
        if 0 <= j < N and cost[i, j] < INF_COST:
            tid = prev_tracks[i][0]
            new_tracks[j] = (tid, current_boxes[j])

    for j in range(N):
        if new_tracks[j] is None:
            new_tracks[j] = (next_id, current_boxes[j])
            next_id += 1

    return new_tracks, next_id


# Distinct BGR palette for per-ID colouring in tracking visualisations.
_TRACK_PALETTE = [
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 0, 128),
    (128, 255, 0),
    (255, 128, 0),
    (0, 0, 255),
    (0, 255, 128),
    (128, 0, 255),
    (255, 0, 64),
    (64, 255, 192),
    (192, 64, 255),
    (255, 192, 64),
    (64, 192, 255),
]


def _track_color(track_id):
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


def synthesize_pan_sequence(img_bgr, num_frames=30, window_ratio=(2 / 3, 2 / 3)):
    """
    Generates a synthetic camera-pan sequence from a single broadcast frame.
    A viewport of size (window_ratio * H, window_ratio * W) slides linearly
    across the source image from left to right (and with a mild vertical
    sinusoid) over `num_frames` steps. This produces a controlled sequence
    where every player's image-plane position changes smoothly between
    frames, which is the regime tracking algorithms are designed for. The
    function yields (frame_index, frame_bgr, (x0, y0)) tuples, where the
    offset (x0, y0) maps a viewport coordinate back to the source frame.
    """
    H, W = img_bgr.shape[:2]
    wh = max(64, int(round(W * window_ratio[1])))
    hh = max(64, int(round(H * window_ratio[0])))
    x_range = max(1, W - wh)
    y_amp = max(0, (H - hh) // 2)
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        x0 = int(round(t * x_range))
        y0 = int(round(y_amp + 0.4 * y_amp * np.sin(2 * np.pi * t)))
        y0 = max(0, min(H - hh, y0))
        frame = img_bgr[y0 : y0 + hh, x0 : x0 + wh].copy()
        yield i, frame, (x0, y0)


def run_tracking_demo(
    image_path,
    save_dir,
    num_frames=30,
    score_threshold=0.6,
    fps=8,
    tracker="hungarian",
    max_dist=120,
):
    """
    End-to-end tracking demonstration on a synthesised camera-pan sequence.

    For each frame the viewport is cropped, Mask R-CNN is run to obtain
    person detections, and detections are associated with previous tracks
    using either the greedy or Hungarian tracker. Each frame is annotated
    with a per-ID colour, the centroid trail of the last few frames, and a
    legend stamp; frames are written to disk and stitched into an MP4. A
    final summary image overlays every centroid trajectory on the last
    viewport for at-a-glance ID-persistence inspection.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = load_color(image_path)
    frames_iter = list(synthesize_pan_sequence(img_bgr, num_frames=num_frames))

    prev_tracks = []
    next_id = 0
    tracker_fn = track_players_hungarian if tracker == "hungarian" else track_players
    trails = {}  # tid -> list of (cx, cy) centroids in viewport coordinates
    written_frames = []

    for i, frame, _ in frames_iter:
        _, boxes, _ = segment_players_maskrcnn(frame, score_threshold=score_threshold)
        new_tracks, next_id = tracker_fn(
            prev_tracks, boxes, max_dist=max_dist, next_id=next_id
        )
        prev_tracks = new_tracks

        overlay = frame.copy()
        for tid, (x, y, w, h) in new_tracks:
            colour = _track_color(tid)
            cx, cy = int(x + w / 2), int(y + h / 2)
            trails.setdefault(tid, []).append((cx, cy))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(
                overlay,
                f"#{tid}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )
            pts = trails[tid][-12:]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(overlay, a, b, colour, 2)

        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"Frame {i + 1:02d}/{num_frames}   active IDs: {len(new_tracks)}   "
            f"total IDs seen: {next_id}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        out = save_dir / f"frame_{i:03d}.jpg"
        save_uniform(out, overlay)
        written_frames.append(overlay)

    # Stitch frames to MP4. mp4v is the most portable codec in OpenCV builds.
    if written_frames:
        h, w = written_frames[0].shape[:2]
        video_path = save_dir / "tracking_demo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (w, h))
        for f in written_frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)
        writer.release()

    # Trajectory overlay on the last viewport: shows ID persistence at a glance.
    if written_frames:
        traj = written_frames[-1].copy()
        for tid, pts in trails.items():
            colour = _track_color(tid)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(traj, a, b, colour, 2)
            if pts:
                cv2.circle(traj, pts[0], 4, colour, -1)
                cv2.circle(traj, pts[-1], 6, colour, 2)
        save_uniform(save_dir / "trajectories.jpg", traj)

    print(
        f"Tracking demo: {num_frames} frames, {next_id} unique IDs assigned, "
        f"{sum(1 for v in trails.values() if len(v) >= num_frames // 2)} IDs "
        f"persisted across at least half the sequence."
    )
    print(f"  Frames + video written to: {save_dir}")
    return save_dir


# --- Module 8: Adaptive classical pipeline (K-means field segmentation) ---
def extract_features_kmeans(img_bgr, k=4, sample_step=4):
    """
    Adaptive field segmentation via K-means in BGR space (no green assumption).
    The largest cluster is treated as the playing field; pixels far from its
    centroid (above a per-image percentile threshold) form the foreground mask
    after the same morphological clean-up used by the HSV variant.
    """
    pixels = img_bgr[::sample_step, ::sample_step].reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)

    counts = np.bincount(labels.flatten(), minlength=k)
    field_color = centers[int(np.argmax(counts))]

    diff = img_bgr.astype(np.float32) - field_color
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    thresh = float(np.percentile(dist, 70))
    fg = (dist > thresh).astype(np.uint8) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_small, iterations=2)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    return fg


# --- Module 9: Team classification via jersey-color clustering ---
# Conceptually identical to Assignment 3 (1-D EM/GMM on intensity); here we
# cluster 3-D HSV jersey descriptors with K-means since OpenCV ships a fast
# K-means and we only need hard team labels per box, not soft posteriors.
_TEAM_COLORS_BGR = [
    (0, 0, 255),  # team A: red
    (255, 0, 0),  # team B: blue
    (0, 255, 255),  # referees: yellow
    (255, 0, 255),  # goalkeepers / others: magenta
    (0, 165, 255),  # extra cluster: orange
]


def _jersey_descriptor(img_bgr, bbox):
    """
    HSV color descriptor of the torso region inside a player bounding box.
    The torso band excludes the head (top 20%) and legs (bottom 45%), and the
    horizontal margins drop hands/background. Very dark (shadow) and very
    bright (lines / kit numbers) pixels are masked out before averaging so
    the descriptor reflects the dominant jersey color rather than artefacts.
    Returns a 3-vector in HSV space, or None if the crop is degenerate.
    """
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    x0 = max(0, x + int(0.20 * w))
    x1 = min(W, x + int(0.80 * w))
    y0 = max(0, y + int(0.20 * h))
    y1 = min(H, y + int(0.55 * h))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    crop = img_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    keep = (v > 30) & (v < 240) & (s > 25)
    pixels = hsv[keep] if keep.sum() >= 10 else hsv.reshape(-1, 3)
    return np.mean(pixels, axis=0).astype(np.float32)


def classify_teams(img_bgr, boxes, k=4, attempts=5):
    """
    Cluster player detections into team / role groups by K-means over jersey
    HSV descriptors. K defaults to 4 to accommodate the two outfield teams,
    the referee(s), and the goalkeepers; pass k=2 for the simple foreground-
    foreground split. Returns a list of integer labels aligned with `boxes`
    (-1 for boxes whose descriptor could not be extracted).
    """
    if not boxes:
        return []
    feats, valid_idx = [], []
    for i, b in enumerate(boxes):
        d = _jersey_descriptor(img_bgr, b)
        if d is not None:
            feats.append(d)
            valid_idx.append(i)
    if not feats:
        return [-1] * len(boxes)

    X = np.asarray(feats, dtype=np.float32)
    K = min(k, len(X))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _, lbl, _ = cv2.kmeans(X, K, None, crit, attempts, cv2.KMEANS_PP_CENTERS)

    out = [-1] * len(boxes)
    for j, i in enumerate(valid_idx):
        out[i] = int(lbl[j, 0])
    return out


def draw_team_detections(img_bgr, boxes, team_labels, thickness=2):
    """Render bounding boxes coloured by team-cluster label (legend cycles)."""
    result = img_bgr.copy()
    for (x, y, w, h), lab in zip(boxes, team_labels):
        color = (
            _TEAM_COLORS_BGR[lab % len(_TEAM_COLORS_BGR)]
            if lab >= 0
            else (180, 180, 180)
        )
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            result,
            f"T{lab}" if lab >= 0 else "?",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return result


# --- Pipeline runners (uniform interface) ---
# Each runner returns a dict: {"name", "mask", "boxes", "scores", "elapsed"}.
_PIPELINE_COLORS = {
    "classical-hsv": (0, 255, 0),
    "classical-kmeans": (0, 200, 255),
    "deeplab": (0, 165, 255),
    "maskrcnn": (255, 100, 50),
}


def _run_classical(img_bgr, method="hsv"):
    t0 = time.perf_counter()
    if method == "kmeans":
        mask = extract_features_kmeans(img_bgr)
    else:
        _, hsv = preprocess_image(img_bgr)
        mask, _ = extract_features(hsv)
    boxes = detect_players(mask)
    return {
        "name": f"classical-{method}",
        "mask": mask,
        "boxes": boxes,
        "scores": [],
        "elapsed": time.perf_counter() - t0,
    }


def _run_deeplab(img_bgr, refine=True):
    t0 = time.perf_counter()
    raw = segment_players_dl(img_bgr)
    mask = refine_dl_mask(raw) if refine else raw
    boxes = mask_to_bboxes(mask)
    return {
        "name": "deeplab",
        "mask": mask,
        "boxes": boxes,
        "scores": [],
        "elapsed": time.perf_counter() - t0,
    }


def _run_maskrcnn(img_bgr, score_threshold=0.6, refine=True):
    t0 = time.perf_counter()
    raw_mask, boxes, scores = segment_players_maskrcnn(
        img_bgr, score_threshold=score_threshold
    )
    mask = refine_dl_mask(raw_mask) if refine else raw_mask
    return {
        "name": "maskrcnn",
        "mask": mask,
        "boxes": boxes,
        "scores": scores,
        "elapsed": time.perf_counter() - t0,
    }


def _evaluate_pipeline_result(result, gt_boxes, gt_mask):
    """Computes Precision/Recall/F1, mean bbox IoU, and mask IoU for one pipeline."""
    p, r, f1 = evaluate_detections(result["boxes"], gt_boxes)
    bbox_iou_mean = compute_bbox_iou_mean(result["boxes"], gt_boxes)
    mask_iou = (
        compute_mask_iou(result["mask"], gt_mask)
        if gt_mask is not None
        else float("nan")
    )
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "bbox_iou_mean": bbox_iou_mean,
        "mask_iou": mask_iou,
    }


def _build_overlay(img_bgr, result, gt_boxes, metrics=None):
    """Draws predictions (in pipeline color) and GT (red) onto a copy of the image."""
    color = _PIPELINE_COLORS.get(result["name"], (200, 200, 0))
    overlay = draw_detections(img_bgr, result["boxes"], color=color)
    if gt_boxes:
        overlay = draw_detections(overlay, gt_boxes, color=(0, 0, 255), thickness=1)
    if metrics is not None:
        text = (
            f"{result['name']} P:{metrics['precision']:.2f} "
            f"R:{metrics['recall']:.2f} F1:{metrics['f1']:.2f}"
        )
        cv2.putText(
            overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
    return overlay


def log_failure_case(failures_dir, img_path, overlay, pipeline_name, metrics):
    """Writes the annotated overlay for an image where a pipeline performed poorly."""
    _p, _r, f1, miou = metrics
    failures_dir = Path(failures_dir)
    out = (
        failures_dir
        / f"{img_path.stem}__{pipeline_name}__F1-{f1:.2f}_mIoU-{miou:.2f}.jpg"
    )
    save_uniform(out, overlay)
    return out


# --- Main Execution ---
def process_image(
    img_path,
    gt_json_path=None,
    save_dir=None,
    pipelines=("classical", "deeplab", "maskrcnn"),
    classical_method="hsv",
    score_threshold=0.6,
    refine_dl=True,
    perturbation=None,
    track_state=None,
    failure_f1_threshold=0.5,
    failure_iou_threshold=0.3,
    classify_teams_k=0,
    show=True,
):
    """
    Runs the requested subset of pipelines on a single image, evaluates each
    against ground truth, optionally logs failure cases, and shows a side-by-side
    comparison. Returns a list of per-pipeline evaluation records (only when GT
    boxes were available for this image).
    """
    tag = f" [{perturbation}]" if perturbation else ""
    print(f"\n--- Processing {img_path.name}{tag} ---")

    img_bgr = load_color(img_path)
    if perturbation:
        img_bgr = apply_perturbations(img_bgr, perturbation)
    H, W = img_bgr.shape[:2]

    # Run each requested pipeline; skip DL ones gracefully if torch is missing.
    results = []
    for pname in pipelines:
        try:
            if pname == "classical":
                results.append(_run_classical(img_bgr, method=classical_method))
            elif pname == "deeplab":
                results.append(_run_deeplab(img_bgr, refine=refine_dl))
            elif pname == "maskrcnn":
                results.append(
                    _run_maskrcnn(
                        img_bgr, score_threshold=score_threshold, refine=refine_dl
                    )
                )
            else:
                print(f"  Unknown pipeline '{pname}', skipping.")
        except ImportError as e:
            print(f"  Pipeline '{pname}' skipped (missing dependency): {e}")
            print("  Install with: pip install torch torchvision")

    # Ground truth (per-image) and pseudo mask for mask-IoU.
    gt_boxes = []
    gt_mask = None
    if gt_json_path:
        gt_boxes = parse_ground_truth(gt_json_path, img_path.name)
        if gt_boxes:
            gt_mask = gt_boxes_to_mask(gt_boxes, H, W)

    panels = [img_bgr]
    titles = ["Original" + (f" ({perturbation})" if perturbation else "")]
    eval_records = []

    for r in results:
        m = (
            _evaluate_pipeline_result(r, gt_boxes, gt_mask)
            if gt_boxes
            else {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "bbox_iou_mean": 0.0,
                "mask_iou": float("nan"),
            }
        )
        overlay = _build_overlay(img_bgr, r, gt_boxes, m if gt_boxes else None)

        panels.append(r["mask"])
        titles.append(f"{r['name']} mask")
        panels.append(overlay)
        titles.append(f"{r['name']} det ({len(r['boxes'])})")

        if gt_boxes:
            print(f"\n--- {r['name']} ---")
            print(f"  Precision:      {m['precision']:.4f}")
            print(f"  Recall:         {m['recall']:.4f}")
            print(f"  F1:             {m['f1']:.4f}")
            print(f"  Mean Bbox IoU:  {m['bbox_iou_mean']:.4f}")
            print(
                f"  Mask IoU*:      {m['mask_iou']:.4f}  (* lower bound; GT is rectangular)"
            )
            print(f"  Time:           {r['elapsed']*1000:.1f} ms")
            eval_records.append(
                {
                    "name": r["name"],
                    "elapsed": r["elapsed"],
                    "n_boxes": len(r["boxes"]),
                    "perturbation": perturbation or "original",
                    **m,
                }
            )

            # Failure-case logging (under <save-dir>/failures/)
            if save_dir is not None and (
                m["f1"] < failure_f1_threshold
                or m["bbox_iou_mean"] < failure_iou_threshold
            ):
                log_failure_case(
                    Path(save_dir) / "failures",
                    img_path,
                    overlay,
                    r["name"] + (f"-{perturbation}" if perturbation else ""),
                    (m["precision"], m["recall"], m["f1"], m["bbox_iou_mean"]),
                )

        # Persist per-pipeline outputs at a uniform width so the dataset
        # produces visually comparable detection figures.
        if save_dir is not None:
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            suffix = f"-{perturbation}" if perturbation else ""
            save_uniform(
                sd / f"{img_path.stem}_{r['name']}{suffix}_mask.png", r["mask"]
            )
            save_uniform(sd / f"{img_path.stem}_{r['name']}{suffix}_det.jpg", overlay)

    if not gt_boxes:
        if gt_json_path:
            print(f"Warning: No ground truth entries found for {img_path.name}")
        else:
            print("Evaluation skipped: No ground truth JSON provided.")

    # Optional centroid tracking on the most accurate available pipeline.
    if track_state is not None and results:
        priority = ["maskrcnn", "deeplab", "classical-hsv", "classical-kmeans"]
        primary = next(
            (r for n in priority for r in results if r["name"] == n), results[0]
        )
        tracker_fn = (
            track_players_hungarian
            if track_state.get("tracker") == "hungarian"
            else track_players
        )
        new_tracks, next_id = tracker_fn(
            track_state["prev"],
            primary["boxes"],
            next_id=track_state["next_id"],
        )
        track_state["prev"] = new_tracks
        track_state["next_id"] = next_id

        track_img = img_bgr.copy()
        for tid, (x, y, w, h) in new_tracks:
            cv2.rectangle(track_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                track_img,
                f"#{tid}",
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
        panels.append(track_img)
        titles.append(f"Tracking ({len(new_tracks)} ids)")

    # Optional team classification on the most accurate available pipeline.
    if classify_teams_k and classify_teams_k > 1 and results:
        priority = ["maskrcnn", "deeplab", "classical-hsv", "classical-kmeans"]
        primary = next(
            (r for n in priority for r in results if r["name"] == n), results[0]
        )
        team_labels = classify_teams(img_bgr, primary["boxes"], k=classify_teams_k)
        team_img = draw_team_detections(img_bgr, primary["boxes"], team_labels)
        panels.append(team_img)
        titles.append(f"Teams (K={classify_teams_k}, on {primary['name']})")
        if save_dir is not None:
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            suffix = f"-{perturbation}" if perturbation else ""
            save_uniform(sd / f"{img_path.stem}_teams{suffix}.jpg", team_img)

    if show:
        show_images(panels, titles, cols=4)

    return eval_records


def _print_final_summary(records, perturbations):
    """Aggregates per-image evaluation records into a per-pipeline summary."""
    if not records:
        print("\n(No ground-truth-evaluated records to summarise.)")
        return

    print("\n========== FINAL SUMMARY ==========")
    by_key = {}
    for rec in records:
        by_key.setdefault((rec["name"], rec["perturbation"]), []).append(rec)

    def _mean_std(vals, reducer=np.mean):
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return float("nan"), float("nan")
        # ddof=1 -> sample standard deviation; n=1 gracefully returns nan.
        m = float(reducer(arr))
        s = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan")
        return m, s

    for (name, pert), recs in sorted(by_key.items()):
        p_m, p_s = _mean_std([r["precision"] for r in recs])
        r_m, r_s = _mean_std([r["recall"] for r in recs])
        f_m, f_s = _mean_std([r["f1"] for r in recs])
        bi_m, bi_s = _mean_std([r["bbox_iou_mean"] for r in recs])
        mi_m, mi_s = _mean_std([r["mask_iou"] for r in recs], reducer=np.nanmean)
        t_m, t_s = _mean_std([r["elapsed"] * 1000 for r in recs])
        N = len(recs)
        print(f"\n{name} [{pert}] (N={N} images)")
        print(f"  Precision:     {p_m:.4f} +/- {p_s:.4f}")
        print(f"  Recall:        {r_m:.4f} +/- {r_s:.4f}")
        print(f"  F1:            {f_m:.4f} +/- {f_s:.4f}")
        print(f"  Mean Bbox IoU: {bi_m:.4f} +/- {bi_s:.4f}")
        print(f"  Mask IoU*:     {mi_m:.4f} +/- {mi_s:.4f}")
        print(f"  Time (ms):     {t_m:.1f} +/- {t_s:.1f}")
    print(
        "\n* Mask IoU is a lower bound for instance segmenters: ground truth is\n"
        "  rectangular, so even a perfect player silhouette fills only ~50-60%\n"
        "  of its GT box. Compare detectors against each other, not against 1.0."
    )

    # Conclusion on the unperturbed images, when comparing >=2 pipelines.
    originals = [r for r in records if r["perturbation"] == "original"]
    by_pipe = {}
    for r in originals:
        by_pipe.setdefault(r["name"], []).append(r)
    if len(by_pipe) >= 2:
        scored = [
            (
                n,
                float(np.mean([r["f1"] for r in rs])),
                float(np.mean([r["elapsed"] for r in rs])),
            )
            for n, rs in by_pipe.items()
        ]
        best = max(scored, key=lambda x: x[1])
        fastest = min(scored, key=lambda x: x[2])
        print("\n--- Conclusion (original images) ---")
        print(f"  Best F1:   {best[0]} (F1={best[1]:.4f})")
        print(f"  Fastest:   {fastest[0]} ({fastest[2]*1000:.1f} ms/image)")
        if best[0] != fastest[0]:
            print(
                f"  Trade-off: {best[0]} is more accurate while "
                f"{fastest[0]} is faster -- classical pipelines are assumption-based "
                f"but quick, while DL pipelines are data-driven but slower."
            )

    # Robustness commentary
    if any(p != "original" for p in perturbations):
        print("\n--- Robustness ---")
        for name in by_pipe.keys():
            base = float(
                np.mean(
                    [
                        r["f1"]
                        for r in records
                        if r["name"] == name and r["perturbation"] == "original"
                    ]
                )
            )
            for pert in perturbations:
                if pert == "original":
                    continue
                pf1 = [
                    r["f1"]
                    for r in records
                    if r["name"] == name and r["perturbation"] == pert
                ]
                if not pf1:
                    continue
                drop = base - float(np.mean(pf1))
                print(f"  {name:20s} F1 drop under '{pert}': {drop:+.4f}")


def _write_csv_report(rows, csv_path):
    import csv

    fieldnames = [
        "image",
        "perturbation",
        "name",
        "precision",
        "recall",
        "f1",
        "bbox_iou_mean",
        "mask_iou",
        "elapsed",
        "n_boxes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nCSV report written to: {csv_path}")


def main():
    import argparse

    default_data = Path(__file__).resolve().parents[1] / "data" / "final-project"
    parser = argparse.ArgumentParser(
        description=(
            "Football player detection: classical CV vs DeepLabV3 vs Mask R-CNN."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help="Directory containing input images (default: %(default)s).",
    )
    parser.add_argument(
        "--gt-json",
        type=Path,
        default=None,
        help="Path to ground_truth.json (default: <data-dir>/ground_truth.json).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to write outputs, masks, and a 'failures/' subfolder.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N images."
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["classical", "deeplab", "maskrcnn"],
        choices=["classical", "deeplab", "maskrcnn"],
        help="Subset of pipelines to run (default: all three).",
    )
    parser.add_argument(
        "--classical-method",
        choices=["hsv", "kmeans"],
        default="hsv",
        help="Classical field-segmentation method.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.6,
        help="Mask R-CNN confidence threshold (default: 0.6).",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable post-processing (closing + small-blob removal) on DL masks.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Also evaluate each image under bright/dark/noise/blur perturbations.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Run tracking across the (sorted) image sequence.",
    )
    parser.add_argument(
        "--tracker",
        choices=["greedy", "hungarian"],
        default="hungarian",
        help=(
            "Tracker association strategy. 'greedy' is nearest-centroid; "
            "'hungarian' is global-optimal assignment over a "
            "(1-IoU)+normalized-distance cost matrix (default)."
        ),
    )
    parser.add_argument(
        "--classify-teams",
        type=int,
        default=0,
        metavar="K",
        help=(
            "If > 1, cluster detections of the most accurate pipeline into K "
            "team/role groups via K-means on jersey HSV descriptors (default: "
            "0 = disabled). Try K=4 for two teams + referee + goalkeepers."
        ),
    )
    parser.add_argument(
        "--pr-curve",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Compute and plot dataset-level Pascal-VOC PR curves for every "
            "enabled pipeline. Mask R-CNN uses native model confidence; "
            "DeepLab uses mean person-class probability per blob; classical "
            "CV uses bbox fill-ratio. Writes a single PNG at PATH."
        ),
    )
    parser.add_argument(
        "--csv-report",
        type=Path,
        default=None,
        help="Write per-image evaluation records as CSV.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the interactive matplotlib display (useful for headless runs).",
    )
    parser.add_argument(
        "--tracking-demo",
        type=Path,
        default=None,
        metavar="IMAGE",
        help=(
            "Run the tracking demo only: synthesises a camera-pan sequence "
            "from IMAGE, runs Mask R-CNN per frame, threads detections "
            "through the selected tracker, and writes annotated frames + an "
            "MP4 + a trajectory overlay. Skips the dataset-level evaluation."
        ),
    )
    parser.add_argument(
        "--tracking-demo-frames",
        type=int,
        default=30,
        help="Number of frames in the synthesised tracking sequence (default 30).",
    )
    parser.add_argument(
        "--tracking-demo-out",
        type=Path,
        default=Path("final-project/results/tracking_demo"),
        help="Output directory for tracking-demo frames and video.",
    )
    args = parser.parse_args()

    if args.tracking_demo is not None:
        run_tracking_demo(
            args.tracking_demo,
            args.tracking_demo_out,
            num_frames=args.tracking_demo_frames,
            score_threshold=args.score_threshold,
            tracker=args.tracker,
        )
        return

    base = args.data_dir
    gt_json = args.gt_json if args.gt_json is not None else base / "ground_truth.json"

    images = sorted(
        p for p in base.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        print(f"No images found in {base}")
        print("Please add dataset images to test the pipeline.")
        return

    print(f"Found {len(images)} image(s) to process.")

    perturbations = ["original"]
    if args.robustness:
        perturbations += ["bright", "dark", "noise", "blur"]

    track_state = (
        {"prev": [], "next_id": 0, "tracker": args.tracker} if args.track else None
    )
    all_records = []
    csv_rows = []

    for img_path in images:
        for pert in perturbations:
            recs = process_image(
                img_path,
                gt_json_path=gt_json if gt_json.exists() else None,
                save_dir=args.save_dir,
                pipelines=tuple(args.pipelines),
                classical_method=args.classical_method,
                score_threshold=args.score_threshold,
                refine_dl=not args.no_refine,
                classify_teams_k=args.classify_teams,
                perturbation=None if pert == "original" else pert,
                track_state=track_state if pert == "original" else None,
                show=not args.no_show,
            )
            for r in recs:
                all_records.append(r)
                csv_rows.append({"image": img_path.name, **r})

    _print_final_summary(all_records, perturbations)

    if args.csv_report:
        _write_csv_report(csv_rows, args.csv_report)

    # Dataset-level Pascal-VOC PR curves + AP for every enabled pipeline.
    # Mask R-CNN uses its native per-instance confidence; the other two
    # pipelines have no confidence and are ranked by proxy scores (mean
    # person-probability for DeepLab, bbox fill-ratio for classical CV).
    if args.pr_curve is not None and gt_json.exists():
        print("\n--- Building dataset-level PR curves ---")
        pipeline_specs = []
        if "maskrcnn" in args.pipelines:
            pipeline_specs.append(
                (
                    "maskrcnn",
                    lambda im: harvest_maskrcnn_detections(im, low_score=0.05),
                    "#1f77b4",
                    "model conf.",
                )
            )
        if "deeplab" in args.pipelines:
            pipeline_specs.append(
                (
                    "deeplab",
                    lambda im: harvest_deeplab_detections(
                        im, refine=not args.no_refine
                    ),
                    "#ff7f0e",
                    "mean person prob.",
                )
            )
        if "classical" in args.pipelines:
            pipeline_specs.append(
                (
                    "classical-" + args.classical_method,
                    lambda im, m=args.classical_method: harvest_classical_detections(
                        im, method=m
                    ),
                    "#2ca02c",
                    "bbox fill ratio",
                )
            )

        gt_by_image = {}
        for img_path in images:
            gt_boxes = parse_ground_truth(gt_json, img_path.name)
            if gt_boxes:
                gt_by_image[img_path.name] = gt_boxes

        curves = []
        for name, harvest_fn, color, score_kind in pipeline_specs:
            dets_by_image = {}
            try:
                for img_path in images:
                    if img_path.name not in gt_by_image:
                        continue
                    img_bgr = load_color(img_path)
                    dets_by_image[img_path.name] = harvest_fn(img_bgr)
            except ImportError as e:
                print(f"  Skipping {name} PR curve (missing dependency): {e}")
                continue
            if not dets_by_image:
                continue
            precision, recall, ap = compute_pr_curve_voc(
                dets_by_image, gt_by_image, iou_threshold=0.5
            )
            n_dets = sum(len(v) for v in dets_by_image.values())
            n_gt = sum(len(v) for v in gt_by_image.values())
            print(
                f"  {name:22s} AP@0.5 = {ap:.4f}  "
                f"({n_dets} dets vs {n_gt} GT across {len(dets_by_image)} images)"
            )
            curves.append(
                {
                    "name": name,
                    "precision": precision,
                    "recall": recall,
                    "ap": ap,
                    "color": color,
                    "score_kind": score_kind,
                }
            )

        if curves:
            out_png = plot_pr_curves(curves, args.pr_curve)
            print(f"  PR plot saved to: {out_png}")


if __name__ == "__main__":
    main()
