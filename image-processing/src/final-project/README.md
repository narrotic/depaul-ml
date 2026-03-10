Below is a **clean, professional `README.md`** tailored exactly to your project structure. It explains the folders, how to run the final project, and satisfies your professor’s requirements.

You can **copy this directly**.

---

# README.md

```markdown
# Image Processing Course Projects

Author: Asad Khan  
Course: Image Processing  
University: DePaul University  

This project implements a computer vision pipeline to detect football players in broadcast match images using classical image processing methods.

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

## Football Player Detection using Image Processing

The final project implements a computer vision pipeline to detect football players in broadcast match images using classical image processing methods.

The system identifies players by separating the green football field from foreground objects and detecting player regions using connected component analysis.

The project also includes convert_annotations.py to convert COCO annotations to the format expected by the pipeline.

This convert_annotations.py script filters annotations to only the images that exist in the images_folder (e.g I had the first 20 images from the dataset and this script will only convert the annotations for those 20 images), then saves a simplified ground_truth.json in the format expected by the pipeline:
    {"image_name.jpg": [[x, y, w, h], ...], ...}

---

# Image Processing Pipeline

The detection pipeline includes the following steps:

1. Image preprocessing
2. Color space conversion (RGB → HSV)
3. Green field segmentation
4. Foreground mask extraction
5. Morphological operations to remove noise
6. Connected component detection
7. Bounding box generation
8. Evaluation using IoU, Precision, Recall, and F1 Score

---

# Dependencies

All dependencies are listed in `requirements.txt`.

Install them with:

```

pip install -r requirements.txt

```

Main libraries used:

- OpenCV
- NumPy
- Matplotlib
- tqdm

---

# Running the Final Project

Navigate to the final project directory:

```

cd src/final-project

```

Then run the main script:

```

python final-project.py

```

The script will:

1. Load images from `data/final-project`
2. Apply the player detection pipeline
3. Generate detection results
4. Compute evaluation metrics for each image

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

The project evaluates detection performance using standard object detection metrics:

- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score

These metrics compare predicted bounding boxes with the ground truth annotations.
The most important metric is the F1 Score, which is the harmonic mean of precision and recall. It is a single metric that balances both precision and recall, and is a good metric to use when comparing different models. Precision measures the accuracy of the positive predictions, and recall measures the ability of the model to find all the positive samples.
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