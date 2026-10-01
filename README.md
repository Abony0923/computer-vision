# Robustness and Failure Threshold Analysis of Local Feature Detection Algorithms Under Image Degradation and Enhancement

## Overview

This project presents a controlled experimental study of the robustness of local feature detection algorithms under different types and levels of image degradation.

Four widely used feature detectors are evaluated:

- **Harris**
- **SIFT**
- **ORB**
- **AKAZE**

The experiments investigate how their feature detection and matching performance changes as image quality progressively deteriorates due to **Gaussian noise, Gaussian blur, and reduced illumination**.

The study also examines whether common image enhancement techniques can recover feature-matching performance on degraded images.

---

## Objectives

The main objectives of this project are to:

- Analyze the robustness of Harris, SIFT, ORB, and AKAZE under different image degradation conditions.
- Identify the **failure threshold** of each detector.
- Compare detector performance under progressively increasing degradation.
- Evaluate the effect of image enhancement on degraded images.
- Determine whether enhancement can extend the reliable operating range of feature detectors.
- Analyze whether the effectiveness of enhancement depends on the type of degradation.

---

## Dataset

The experiments use the **HPatches dataset**, a benchmark dataset designed for evaluating local feature detection and image matching.

A subset of **20 sequences** is used in this study:

- 10 illumination sequences
- 10 viewpoint sequences

The dataset provides reference images, transformed images, and ground-truth homographies, allowing feature matches to be evaluated geometrically.

Dataset: https://hpatches.github.io/

---

## Feature Detectors

### Harris

Harris Corner Detector is a classical corner detection method that identifies points with significant intensity changes in multiple directions.

### SIFT

Scale-Invariant Feature Transform (SIFT) detects and describes distinctive local features while providing robustness to scale and rotation changes.

### ORB

Oriented FAST and Rotated BRIEF (ORB) combines FAST keypoint detection with a binary BRIEF-based descriptor, providing an efficient alternative for feature matching.

### AKAZE

AKAZE detects features in a nonlinear scale space and uses efficient binary descriptors for matching.

---

## Image Degradation

Three types of degradation are investigated.

### 1. Gaussian Noise

Random Gaussian noise is added progressively to simulate sensor noise and other noisy imaging conditions.

### 2. Gaussian Blur

Gaussian filtering is progressively increased to simulate loss of image sharpness caused by blur.

### 3. Reduced Illumination

Image brightness is progressively reduced to simulate low-light imaging conditions.

---

## Image Enhancement

Three enhancement techniques are applied to degraded images.

### Global Histogram Equalization

Improves global image contrast by redistributing pixel intensities.

### CLAHE

Contrast-Limited Adaptive Histogram Equalization improves local contrast while limiting excessive amplification of noise.

### Gamma Correction

Gamma correction is used to adjust image brightness, particularly for recovering visibility in dark images.

---

## Evaluation Metrics

The detectors are evaluated using several quantitative measures:

- **Keypoint Count** — number of detected feature points.
- **Correct-Match Retention** — percentage of correct matches retained relative to the clean-image baseline.
- **Matching Accuracy** — proportion of matches considered geometrically correct.
- **Geometric Error** — error associated with the estimated transformation.
- **Failure Threshold** — degradation level at which correct-match retention falls below 50% of the clean-image baseline.

---

## Experimental Pipeline

```text
                HPatches Dataset
                       │
                       ▼
                Clean Images
                       │
                       ▼
             Feature Detection
          ┌────────┬────────┬────────┬────────┐
          ▼        ▼        ▼        ▼
       Harris    SIFT      ORB      AKAZE
          │        │        │        │
          └────────┴────────┴────────┘
                       │
                       ▼
              Progressive Degradation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Gaussian     Gaussian      Reduced
        Noise        Blur        Illumination
          │            │            │
          └────────────┼────────────┘
                       ▼
                Feature Matching
                       │
                       ▼
               Performance Analysis
                       │
                       ▼
                Failure Threshold
                       │
                       ▼
               Image Enhancement
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Histogram     CLAHE       Gamma
       Equalization             Correction
          │            │            │
          └────────────┼────────────┘
                       ▼
              Final Performance
                  Comparison
```

---

## Key Findings

The experiments reveal that different feature detectors respond differently to different degradation types.

- **SIFT** reaches its failure threshold relatively early under Gaussian noise and Gaussian blur.
- **AKAZE** demonstrates strong tolerance to both noise and blur.
- **Harris** shows strong resilience under reduced illumination but is particularly sensitive to Gaussian noise.
- Image enhancement provides more noticeable recovery under **reduced illumination**.
- Enhancement provides limited or inconsistent improvement under **Gaussian noise and blur**.
- The effectiveness of enhancement depends strongly on the type and severity of degradation.

These observations highlight that there is no single preprocessing technique that universally improves feature detection under all degraded imaging conditions.

---

## Technologies Used

- **Python**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **HPatches Dataset**

## Running the Project

Run the main Python script:

```bash
python main.py
```

The program performs the feature detection, degradation, matching, evaluation, and analysis steps defined in the experimental framework.

---

## Research Contribution

This project focuses specifically on identifying **when** feature detectors become unreliable rather than only comparing their performance at a few predefined degradation levels.

By measuring detector-specific failure thresholds and evaluating enhancement techniques under the same controlled conditions, the study provides a quantitative view of feature detector robustness across different degradation types.

---

## Authors

**Maharin Sharif Abony**  
Department of Computer Science and Engineering  
BRAC University  
Dhaka, Bangladesh

**Nazib Uddin Ahrar**  
Department of Computer Science and Engineering  
BRAC University  
Dhaka, Bangladesh

---

## License

This project is intended for academic and research purposes.
