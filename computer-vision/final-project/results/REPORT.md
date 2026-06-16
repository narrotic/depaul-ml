# Football Player Detection — Evaluation Report

**Author:** Asad Khan  
**Course:** CSC528 — Computer Vision, DePaul University  
**Dataset:** 20 broadcast football frames, 293 ground-truth player boxes (COCO-format, filtered)  
**Test image resolution:** native; saved figures normalised to 1280 px wide via `save_uniform`.

## 1. Experimental Setup

Three pipelines were evaluated on the same dataset under identical ground truth:

| Pipeline | Family | Backbone / Method | Weights |
|---|---|---|---|
| `classical-hsv` | Classical CV | HSV field segmentation + morphology + connected components | none |
| `deeplab` | Semantic segmentation (DL) | DeepLabV3 + ResNet-50 | COCO / VOC pretrained |
| `maskrcnn` | Instance segmentation (DL) | Mask R-CNN + ResNet-50-FPN | COCO pretrained |

Detection bookkeeping is class-agnostic at the metric layer; the DL pipelines are filtered to the `person` class before evaluation. The Mask R-CNN confidence threshold during the main run is 0.6.

## 2. Quantitative Evaluation (N = 20 images)

All numbers are mean ± sample standard deviation (ddof = 1) across the 20 images. `Mask IoU*` is computed against a rectangular pseudo-GT mask built from the bounding boxes; for instance segmenters it is a **lower bound** because tight player silhouettes only fill ~50–60 % of each GT rectangle.

| Pipeline | Precision | Recall | F1 | Mean Bbox IoU | Mask IoU* | Time (ms) |
|---|---|---|---|---|---|---|
| `classical-hsv` | 0.743 ± 0.094 | 0.617 ± 0.126 | 0.668 ± 0.093 | 0.569 ± 0.107 | 0.083 ± 0.009 | **9.2 ± 4.7** |
| `deeplab` | 0.611 ± 0.092 | 0.912 ± 0.108 | 0.727 ± 0.083 | 0.851 ± 0.074 | 0.161 ± 0.029 | 2685 ± 609 |
| `maskrcnn` | **0.915 ± 0.063** | **0.973 ± 0.039** | **0.942 ± 0.037** | **0.866 ± 0.032** | **0.517 ± 0.050** | 945 ± 301 |

### Dataset-Level Average Precision

A Pascal-VOC all-points interpolated PR curve was computed for **every** pipeline by pooling all detections across the 20 images and sweeping score thresholds. Mask R-CNN uses its native per-instance confidence; the other two pipelines emit no confidence and are ranked by proxy signals (mean person-class softmax probability per blob for DeepLab; bounding-box fill-ratio — foreground pixels / box area — for classical HSV). The ranking signal is shown in the plot legend.

| Detector | Score signal | Pooled detections | GT boxes | **AP@0.5** |
|---|---|---|---|---|
| Mask R-CNN (person) | native model confidence | 887 | 293 | **0.981** |
| DeepLabV3 (person) | mean person prob. per blob | 514 | 293 | 0.899 |
| Classical HSV | bbox fill ratio | 244 | 293 | 0.582 |

The AP ordering matches the F1 ordering from §2, confirming that the proxy ranking signals are monotonic in detection quality. PR plot: `pr_curve.png` (three curves on one axis with iso-F1 = 0.5 reference).

### Per-Pipeline F1 Range

| Pipeline | min F1 | median F1 | max F1 | hardest image | easiest image |
|---|---|---|---|---|---|
| `classical-hsv` | 0.462 | 0.667 | 0.857 | `13.jpg` (F1=0.46) | `3.jpg` (F1=0.86) |
| `deeplab` | 0.541 | 0.743 | 0.865 | `14.jpg` (F1=0.54) | `7.jpg` (F1=0.87) |
| `maskrcnn` | 0.857 | 0.951 | 1.000 | `11.jpg` (F1=0.86) | `1.jpg` (F1=1.00) |

## 3. Visualization Outputs

All figures are saved under `final-project/results/`:

- **PR curve:** `pr_curve.png` — high-DPI plot, professional palette, includes iso-F1 = 0.5 reference contour and shaded AUC.
- **Detection overlays:** `<image>_<pipeline>_det.jpg` — predictions colour-coded by pipeline, GT in red, per-image P/R/F1 stamped in the upper-left corner.
- **Binary masks:** `<image>_<pipeline>_mask.png` — the foreground mask each pipeline produced before bounding-box extraction.
- **Team classification:** `<image>_teams.jpg` — Mask R-CNN detections recoloured by K-means clusters (K = 4) over HSV jersey descriptors (torso band).
- **Failure cases:** `failures/<image>__<pipeline>__F1-..._mIoU-....jpg` — only `13.jpg` (`classical-hsv`, F1 = 0.46) triggered the failure threshold; Mask R-CNN produced no failure cases.
- **Raw CSV:** `report.csv` — per-image record (image, pipeline, P, R, F1, bbox IoU, mask IoU, runtime, n_boxes).
- **Stdout log:** `run.log` — full console transcript.

All raster outputs are saved at a canonical 1280 px width via the `save_uniform` helper so the **object-to-frame ratio is constant across the dataset**, which is essential for fair side-by-side inspection.

## 4. Observations

### 4.1 Performance Consistency

Mask R-CNN is the only pipeline whose worst-case F1 (0.857 on `11.jpg`) stays above the other two pipelines' *median* F1. Its standard deviation in F1 is 0.037 — roughly 2.5× tighter than the classical (0.093) and DeepLab (0.083) pipelines, indicating substantially more stable performance across both controlled and crowded frames. In particular, the gap between best-case and worst-case F1 collapses from 0.40 (classical) and 0.32 (DeepLab) to just 0.14 for Mask R-CNN.

`13.jpg` and `14.jpg` are the hardest images for the non-instance pipelines, where occluded clusters of overlapping players cause the classical foreground mask and the DeepLab semantic mask to merge multiple individuals into a single connected component. Mask R-CNN side-steps this entirely because its detection head produces a separate proposal per instance before mask refinement, so adjacent players stay separated even when their silhouettes touch.

DeepLab achieves the highest recall after Mask R-CNN (0.91) but has the worst precision (0.61). The recall is high because the semantic `person` mask reliably covers every player; the precision is low because two physically distinct players in contact get merged into one connected component when the mask is converted to bounding boxes, producing a single wide false-positive-style box that fails the IoU = 0.5 match.

### 4.2 Computational Trade-offs

The latency picture is the inverse of the accuracy picture:

| Pipeline | F1 | Mean latency | Cost per F1 point gained over classical |
|---|---|---|---|
| `classical-hsv` | 0.668 | 9 ms | — |
| `deeplab` | 0.727 | 2685 ms | ≈ **45 000 ms / F1 point** |
| `maskrcnn` | 0.942 | 945 ms | ≈ **3 400 ms / F1 point** |

Mask R-CNN is roughly **100× slower** than the classical pipeline in absolute terms, but it is **2.8× faster** than DeepLab while delivering a much larger F1 gain — a per-instance detection head is both more accurate (because it preserves instance boundaries) and computationally cheaper (because the segmentation head only runs on RoI-pooled features inside detected boxes, not on the full image grid). For broadcast-rate inference (25–30 FPS, ≈ 33 ms / frame budget) none of these pipelines is real-time on CPU/MPS; Mask R-CNN would need GPU acceleration and either FP16 inference or a lighter backbone (e.g. ResNet-18-FPN) to fit the budget.

The classical pipeline therefore remains useful as a screening / fall-back stage: at sub-10 ms it can pre-filter empty broadcast frames or trigger the expensive Mask R-CNN only when foreground motion is present.

### 4.3 Sequence-Level Tracking Demonstration

The 20 evaluation images are independent broadcast frames, so per-image tracking IDs cannot be validated against a temporal ground truth. To exercise the Hungarian tracker on a genuine sequence, `--tracking-demo` synthesises a 30-frame camera pan over a single high-resolution broadcast frame (`1.jpg`): a 1280×720 viewport slides from left to right across the 1920×1080 source with a small vertical sinusoid, so every player undergoes smooth image-plane motion between consecutive frames while the underlying scene content is real.

For each synthesised frame Mask R-CNN produces person detections (`score ≥ 0.6`), and the global-optimal Hungarian tracker associates them to the prior frame using

```
c_ij = w · (1 − IoU(b_i, b_j)) + (1 − w) · (d_ij / max_dist),   w = 0.5
```

with a gating distance `max_dist = 120 px`. Outputs in `final-project/results/tracking_demo/`:

- `frame_000.jpg` … `frame_029.jpg` — annotated frames with per-ID colour, centroid trails (last 12 frames), and a header stamp showing active/total ID counts.
- `tracking_demo.mp4` — 8 FPS stitched playback for in-presentation viewing.
- `trajectories.jpg` — every centroid trajectory overlaid on the final frame; long, monotonic trails indicate stable identity, while short stubs are players that entered or exited the viewport during the pan.

**Result.** 30 frames, **34 unique IDs assigned**, **15 IDs persisted across at least half the sequence**. The 34 − 15 = 19 short-lived IDs are predominantly players that entered from the right edge or exited from the left edge as the camera panned 640 px to the right; this is the expected behaviour of any tracker that does not perform re-identification across the field-of-view boundary. The 15 long-lived trails in `trajectories.jpg` confirm that the Hungarian assignment preserves identity through the cluttered mid-field region where greedy nearest-centroid trackers typically swap IDs.

### 4.4 Caveats

1. **Mask IoU is a lower bound** for Mask R-CNN by construction (rectangular GT vs. silhouette prediction). Its absolute value of 0.52 should not be compared against 1.0; against the other pipelines it is already 3.2× higher than DeepLab and 6.2× higher than classical.
2. **N = 20** images yields wide confidence intervals; the 95% CI on the Mask R-CNN F1 mean is roughly ± 0.018, well below the gap to the next-best pipeline, so the ordering is statistically reliable but the absolute numbers should be re-validated on a larger held-out set.
3. The dataset contains no extreme weather or floodlight scenes, so the HSV pipeline's green-grass assumption is never violated; under realistic broadcast diversity its degradation would be substantially worse than reported here.
