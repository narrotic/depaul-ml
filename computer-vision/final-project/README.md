Below is a **clean, professional `README.md`** tailored exactly to your project structure. It explains the folders, how to run the final project, and satisfies your professor’s requirements.

You can **copy this directly**.

---

# README.md

```markdown
# Image Processing Course Projects

Author: Asad Khan  
Course: Image Processing  
University: DePaul University  

This project implements a research-style computer vision system that detects football players in broadcast match images using **three pipelines side-by-side**:

1. A **classical image processing pipeline** with two interchangeable methods:
   - HSV-threshold field segmentation (green-grass assumption)
   - **K-means** field segmentation (no green assumption — adaptive to any dominant pitch color)
2. A **semantic segmentation** pipeline built on **DeepLabV3-ResNet50** (transfer learning, pretrained on COCO with VOC labels).
3. An **instance segmentation** pipeline built on **Mask R-CNN ResNet50-FPN** (transfer learning, pretrained on COCO).

The pipelines run on the same image and are compared visually and quantitatively on:
- Precision / Recall / F1 (bounding-box level, IoU 0.5 threshold)
- **Mean bounding-box IoU** (localisation quality independent of the 0.5 threshold)
- **Mask IoU** vs a pseudo ground-truth mask built from GT boxes
- Runtime (per-image milliseconds)

The system also includes optional **robustness testing** under brightness/dark/noise/blur perturbations, **centroid-distance tracking** across the image sequence, automatic **failure-case logging**, and a **CSV report** export.

---

# Repository Structure

The repository is organized into two main directories:

```

src/final-project/final-project.py     → Final course project implementation
data/final-project/     → Dataset for the final project

```

### src/

Contains the implementation code for homework assignments and the final project.

```

src/final-project/final-project.py     → Final course project implementation

```

### data/

Contains datasets or images required to run each assignment and the final project.

```

data/final-project/     → Dataset for the final project

```

Each assignment folder in `src/` corresponds to a dataset folder in `data/`.

---

# Final Project

## Football Player Detection: Classical CV vs DeepLabV3 vs Mask R-CNN

The final project implements a comparison system across three player-detection pipelines:

- **Classical pipeline** — separates the field from foreground objects and recovers players via connected component analysis. Two field-segmentation methods are available: HSV thresholding (`--classical-method hsv`, default) and K-means clustering (`--classical-method kmeans`, no green assumption).
- **DeepLabV3** (semantic segmentation) — pretrained DeepLabV3-ResNet50, keeps the `person` class mask, then converts the mask into bounding boxes.
- **Mask R-CNN** (instance segmentation) — pretrained Mask R-CNN ResNet50-FPN, filters detections by COCO `person` class and confidence (`--score-threshold`, default 0.6), unions the kept instance masks, and exports both per-instance bounding boxes and the combined mask.

All three pipelines run on every image, are evaluated against the same ground truth, and are compared on detection quality, mask alignment, and runtime. Failed predictions (low F1 or low mean bbox IoU) are automatically saved to `<save-dir>/failures/` for inspection.

The project also includes `convert_annotations.py` to convert COCO annotations to the format expected by the pipeline.

This convert_annotations.py script filters annotations to only the images that exist in the images_folder (e.g I had the first 20 images from the dataset and this script will only convert the annotations for those 20 images), then saves a simplified ground_truth.json in the format expected by the pipeline:
    {"image_name.jpg": [[x, y, w, h], ...], ...}

---

# Detection Pipelines

All three pipelines below run independently for each image and produce a uniform result `{name, mask, boxes, scores, elapsed}` so they can be compared apples-to-apples.

## A. Classical CV Pipeline

Two field-segmentation methods are interchangeable via `--classical-method`:

**HSV (default)**

1. Gaussian blur for noise reduction
2. BGR → HSV conversion
3. Green-field segmentation via HSV thresholding
4. Foreground mask = inverse of green mask
5. Morphological opening + closing
6. Connected components with area + aspect-ratio filters
7. Bounding box generation

**K-means** (`--classical-method kmeans`, no green assumption)

1. Sub-sample pixels and run K-means in BGR space (`k=4`)
2. Pick the **largest cluster** as the field colour
3. Foreground = pixels whose distance from the field colour exceeds the 70th percentile (per-image adaptive threshold)
4. Same morphological clean-up + connected components as the HSV variant

## B. DeepLabV3 Semantic Segmentation (Transfer Learning)

1. Lazy-load **DeepLabV3-ResNet50** (`DeepLabV3_ResNet50_Weights.DEFAULT`, pretrained on COCO with VOC label semantics)
2. Auto-select device: **CUDA → MPS → CPU**
3. BGR → RGB, ImageNet-normalised tensor
4. Forward pass → per-pixel class predictions
5. Keep only the `person` class (VOC index 15) → binary mask
6. **Refinement** (`refine_dl_mask`): morphological closing → small-blob removal → optional Gaussian smoothing
7. Convert refined mask to bounding boxes via `mask_to_bboxes`

## C. Mask R-CNN Instance Segmentation (Transfer Learning)

1. Lazy-load **Mask R-CNN ResNet50-FPN** (`MaskRCNN_ResNet50_FPN_Weights.DEFAULT`, pretrained on COCO)
2. Same auto device selection as above
3. Forward pass → per-instance `(label, score, box, mask_probs)` tuples
4. Filter to **`label == person`** AND **`score >= --score-threshold`** (default `0.6`)
5. Binarise each instance mask at `0.5` and union them into the pipeline's foreground mask
6. Apply the same `refine_dl_mask` post-processing
7. Bounding boxes come **directly from the detector** (per-instance), not from connected components — so adjacent players stay separated

If `torch` is not installed, both DL pipelines are skipped gracefully and the classical pipeline still runs.

---

# Dependencies

All base dependencies are listed in `requirements.txt`.

Install them with:

```
pip install -r requirements.txt
```

Main libraries used:

- OpenCV
- NumPy
- Matplotlib
- tqdm

The deep learning pipeline additionally requires PyTorch and torchvision:

```
pip install torch torchvision
```

On Apple Silicon, the default wheels include MPS support, which the script picks up automatically.

---

# Running the Final Project

Navigate to the final project directory:

```
cd computer-vision/final-project
```

Then run the main script:

```
python final-project.py
```

Available CLI flags (paths are configurable; nothing is hardcoded):

| Flag | Purpose | Default |
|---|---|---|
| `--data-dir` | Directory containing input images | `<repo>/computer-vision/data/final-project` |
| `--gt-json`  | Path to `ground_truth.json` | `<data-dir>/ground_truth.json` |
| `--save-dir` | Directory for annotated outputs, masks, and a `failures/` subfolder | `None` (display only) |
| `--limit`    | Process at most N images (handy for smoke tests) | all images |
| `--pipelines` | Subset of pipelines to run; any combination of `classical`, `deeplab`, `maskrcnn` | all three |
| `--classical-method` | `hsv` or `kmeans` for the classical field-segmentation method | `hsv` |
| `--score-threshold` | Mask R-CNN confidence threshold | `0.6` |
| `--no-refine` | Disable post-processing (closing + small-blob removal) on DL masks | refinement on |
| `--robustness` | Also evaluate each image under `bright` / `dark` / `noise` / `blur` perturbations | off |
| `--track` | Run centroid-distance tracking across the (sorted) image sequence | off |
| `--csv-report` | Write per-image evaluation records as CSV at the given path | none |
| `--no-show` | Skip the interactive matplotlib display (useful for headless / CI runs) | display on |

Examples:

```
# Quick smoke test on the first 3 images, all pipelines
python final-project.py --limit 3

# Classical-only (no torch needed), K-means field segmentation
python final-project.py --pipelines classical --classical-method kmeans

# Full robustness sweep + CSV export, headless
python final-project.py --robustness --csv-report report.csv --save-dir outputs --no-show

# Mask R-CNN only, with tracking and a stricter confidence threshold
python final-project.py --pipelines maskrcnn --score-threshold 0.75 --track

# Run on a custom dataset and write annotated outputs
python final-project.py --data-dir /path/to/images --gt-json /path/to/gt.json --save-dir outputs
```

For each image the script will:

1. Load the image from `--data-dir` (and optionally apply a perturbation under `--robustness`)
2. Run each enabled pipeline; collect `{name, mask, boxes, scores, elapsed}`
3. Display panels for the original image plus, per pipeline, its mask and its detection overlay (GT boxes in red, predictions colour-coded per pipeline)
4. Print per-pipeline metrics (`Precision`, `Recall`, `F1`, `Mean Bbox IoU`, `Mask IoU`, `Time`)
5. Optionally save annotated overlays and binary masks under `--save-dir`, plus failure-case overlays under `<save-dir>/failures/`
6. After all images: print a **final summary** averaging metrics per pipeline (and per perturbation), with an auto-generated **conclusion** (best F1, fastest, trade-off line) and a **robustness** block showing F1 drop per perturbation
7. If `--csv-report` is set, dump every per-image record to CSV (`image, perturbation, name, precision, recall, f1, bbox_iou_mean, mask_iou, elapsed, n_boxes`)

---

# Dataset

The images used for the final project are stored in:

```

data/final-project

```

The dataset consists of football match images used to detect and evaluate player detections.

Ground truth annotations are used to compute evaluation metrics such as precision, recall, and F1 score.

---

# Evaluation Metrics

For each pipeline and each image the project reports:

- **Precision** — accuracy of the positive predictions.
- **Recall** — fraction of ground-truth players that were found.
- **F1 Score** — harmonic mean of precision and recall, the headline metric for comparing pipelines.
- **Mean Bounding-box IoU** — `compute_bbox_iou_mean` averages, over each GT box, the IoU of its best-matching predicted box. Independent of the 0.5 match threshold used by P/R/F1, so it captures pure localisation quality.
- **Mask IoU** — `compute_mask_iou` compares the pipeline's binary foreground mask against a pseudo ground-truth mask built from GT boxes (`gt_boxes_to_mask`). Rewards pipelines whose masks tightly cover the actual player regions; Mask R-CNN typically wins this convincingly because its per-instance masks aren't inflated by full GT bounding boxes.
- **Runtime** — printed per pipeline in milliseconds using `time.perf_counter()`.

Per-image console output (one block per enabled pipeline):

```
--- Processing 0.jpg ---

--- classical-hsv ---
  Precision:      0.7778
  Recall:         0.4118
  F1:             0.5385
  Mean Bbox IoU:  0.3791
  Mask IoU:       0.0819
  Time:           28.4 ms

--- deeplab ---
  Precision:      0.7895
  Recall:         0.8824
  F1:             0.8333
  Mean Bbox IoU:  0.7972
  Mask IoU:       0.1440
  Time:           4682.2 ms

--- maskrcnn ---
  Precision:      0.9375
  Recall:         0.8824
  F1:             0.9091
  Mean Bbox IoU:  0.7872
  Mask IoU:       0.5528
  Time:           10910.8 ms
```

After all images are processed, a **final summary** is printed that aggregates every `(pipeline, perturbation)` group:

```
========== FINAL SUMMARY ==========

classical-hsv [original] (N images)
  Avg Precision:     ...
  Avg Recall:        ...
  Avg F1:            ...
  Avg Mean Bbox IoU: ...
  Avg Mask IoU:      ...
  Avg Time:          ...

deeplab   [original] (N images) ...
maskrcnn  [original] (N images) ...

--- Conclusion (original images) ---
  Best F1:   maskrcnn (F1=0.9091)
  Fastest:   classical-hsv (8.5 ms/image)
  Trade-off: maskrcnn is more accurate while classical-hsv is faster --
             classical pipelines are assumption-based but quick, while
             DL pipelines are data-driven but slower.

--- Robustness ---           # only when --robustness is set
  classical-hsv        F1 drop under 'bright': +0.0123
  classical-hsv        F1 drop under 'dark':   +0.0451
  classical-hsv        F1 drop under 'noise':  +0.1230
  classical-hsv        F1 drop under 'blur':   +0.0734
  maskrcnn             F1 drop under 'bright': +0.0080
  maskrcnn             F1 drop under 'dark':   +0.0210
  ...
```

A positive F1 drop indicates degraded performance under that perturbation; values close to zero indicate a robust pipeline. Empirically Mask R-CNN holds up far better than the classical pipelines under noise and blur, while the HSV pipeline collapses under heavy lighting shifts that destroy its green-grass assumption.

When `--csv-report path.csv` is supplied, the same per-image records are also written to disk for downstream analysis (e.g. plotting with pandas or LaTeX tables in a write-up):

```
image,perturbation,name,precision,recall,f1,bbox_iou_mean,mask_iou,elapsed,n_boxes
0.jpg,original,classical-hsv,0.7778,0.4118,0.5385,0.3791,0.0819,0.0284,7
0.jpg,original,maskrcnn,0.9375,0.8824,0.9091,0.7872,0.5528,10.9108,15
...
```

---

# Notes

- Virtual environments are not included in this repository to keep the submission lightweight.
- Only the required datasets and source code are included.

---

# Author

Asad Khan  
MS Artificial Intelligence  
DePaul University

---