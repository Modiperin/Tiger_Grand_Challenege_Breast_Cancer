# TIGER Grand Challenge

## Tumor Infiltrating Lymphocytes in Breast Cancer

Welcome to the **TIGER Grand Challenge** repository! This project focuses on developing computational algorithms to analyze histopathology whole-slide images (WSIs) of breast cancer tissues. The main objective is to detect Tumor-Infiltrating Lymphocytes (TILs), segment invasive tumors and tumor-associated stroma, and compute an automated TILs score, which is crucial for prognostic and predictive cancer outcomes.

**More detailed information and datasets are available at** [tiger.grand-challenge.org](https://tiger.grand-challenge.org).


## Table of Contents

1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Approach](#approach)
    - [1. Image Preprocessing](#1-image-preprocessing)
    - [2. Cell Detection](#2-cell-detection)
    - [3. Tumor and Stroma Segmentation](#3-tumor-and-stroma-segmentation)
    - [4. TILs Score Calculation](#4-tils-score-calculation)
5. [Model Architectures](#model-architectures)
    - [YOLOv8 for Detection](#yolov8-for-detection)
    - [Efficient-UNet for Segmentation](#efficient-unet-for-segmentation)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Significance of TILs in Breast Cancer](#significance-of-tils-in-breast-cancer)
8. [Results](#results)
9. [Challenges](#challenges)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

The **TIGER Grand Challenge** aims to automate the detection, segmentation, and scoring of Tumor-Infiltrating Lymphocytes (TILs) in breast cancer whole-slide images. TILs are immune cells that infiltrate tumors and serve as key biomarkers in understanding the immune response, prognosis, and response to treatment. This project involves:

1. Detecting lymphocytes and plasma cells within histopathology images.
2. Segmenting invasive tumors and tumor-associated stroma.
3. Computing an automated TILs score based on the detected immune cells and segmented tumor regions.

## Objectives

The primary goals of this project are:
1. **Detection of Lymphocytes and Plasma Cells**: Using object detection models to locate and identify the main immune cells within the tissue.
2. **Segmentation of Invasive Tumor and Tumor-Associated Stroma**: Segmenting the tumor regions and associated stroma to localize key areas for TILs evaluation.
3. **Automated TILs Score Calculation**: Computing a quantitative score that reflects the density of immune cells within tumor-associated stroma, serving as a crucial biomarker.

## Dataset

The dataset used in this challenge comprises high-resolution **Whole Slide Images (WSIs)** of H&E-stained breast cancer tissues, annotated with the following:
- **Lymphocytes and Plasma Cells**: For cell detection.
- **Tumor and Tumor-Associated Stroma**: For tissue segmentation.

### Annotations:

- **Detection Annotations**: Provided in COCO format for immune cells.
- **Segmentation Labels**: For different tissue types:
  - Invasive tumor (label = 1)
  - Tumor-associated stroma (label = 2)
  - In-situ tumor (label = 3)
  - Healthy glands (label = 4)
  - Necrosis not in-situ (label = 5)
  - Inflamed stroma (label = 6)
  - Rest (label = 7)

## Approach

### 1. Image Preprocessing

#### **Patch Creation**
Instead of resizing the images, which could result in a loss of critical details, the WSIs are divided into **512x512 patches**. This patching process helps retain the granular information essential for accurate detection and segmentation of cells and tissue regions.

#### **Color/Stain Normalization**
Histopathology images suffer from variations in color due to different staining methods and equipment. We apply **color normalization** to standardize the appearance of the images, ensuring that color differences do not hinder model performance. This adjustment is particularly important in histopathology for improving diagnostic accuracy.

### 2. Cell Detection

For the detection of lymphocytes and plasma cells, we employ the **YOLOv8** object detection model. The immune cells are detected using a multi-scale anchor-free architecture that allows efficient and accurate localization of small objects such as lymphocytes.

- **Image Size**: The images are resized to **512x512** for detection.
- **Annotations**: The **COCO format** detection annotations are used to train the model.

### 3. Tumor and Stroma Segmentation

We use the **Efficient-UNet** architecture for segmenting invasive tumor regions and tumor-associated stroma. The architecture is based on the U-Net model, combined with EfficientNet as the encoder backbone.

#### **Segmentation Process**:
- **Input Patches**: The model processes **512x512 patches** of the WSIs.
- **Color Normalization**: Helps improve the segmentation performance by minimizing color variations due to different staining methods.
- **Label Assignment**: Each tissue type is assigned a unique label, and the model is trained to segment these regions.

### 4. TILs Score Calculation

Once the lymphocytes and plasma cells are detected, and the invasive tumor and stroma are segmented, we compute the **TILs Score**. This score is calculated by determining the number of detected lymphocytes within the tumor-associated stroma, normalized by the area of the stroma.

```python
def create_til_score(image_path, xml_path, output_path):
    """Compute the TILs score"""
    points = slide_nms(image_path, xml_path, 256)
    wsd_points = to_wsd(points)
    til_area = dist_to_px(8, 0.5) ** 2
    tils_area = len(wsd_points) * til_area
    stroma_area = get_mask_area(TUMOR_STROMA_MASK_PATH)
    tilscore = (100 / int(stroma_area)) * int(tils_area)
    write_json(tilscore, output_path)
```


## Model Architectures

### YOLOv8 for Detection

The YOLOv8 model is used for detecting lymphocytes and plasma cells. It offers:

- **Anchor-Free Detection:** Improves the efficiency and accuracy of detecting small objects like lymphocytes.
- **Multi-Scale Prediction:** Ensures that small cells are detected at different scales, enhancing precision.
- **Efficient:** YOLOv8 is optimized for both speed and accuracy, making it ideal for large datasets like WSIs.

### Efficient-UNet for Segmentation

For segmentation, we used Efficient-UNet, a combination of EfficientNet as the encoder and U-Net as the decoder. This model architecture offers:

- **Efficient Feature Extraction:** Through EfficientNet's pre-trained encoder.
- **Fine-Grained Segmentation:** Using U-Net's skip connections to preserve spatial information.
- **Transfer Learning:** The model leverages pre-trained weights from ImageNet for better feature generalization.

## Evaluation Metrics

We employ the following metrics to evaluate the performance of detection, segmentation, and TILs scoring models:

- **Intersection-over-Union (IoU):** For measuring the overlap between predicted and actual segmentation.
- **Average Precision (AP):** To evaluate the detection accuracy of lymphocytes and plasma cells.
- **TILs Score Correlation:** A quantitative comparison of automated TILs scores with pathologist evaluations and clinical outcomes.

## Significance of TILs in Breast Cancer

Tumor-Infiltrating Lymphocytes (TILs) are critical in breast cancer for:

- **Prognosis:** Higher TIL levels often indicate a better immune response, correlating with improved survival rates.
- **Treatment Prediction:** TILs are predictive markers for the success of immunotherapy and the patient's response to chemotherapy.

## Results

Our approach yielded significant improvements in the detection and segmentation of lymphocytes, plasma cells, invasive tumors, and stroma. The automated TILs score provided a reliable prognostic indicator for immune response evaluation in breast cancer patients.

- **Detection Accuracy:** Achieved high accuracy in detecting lymphocytes and plasma cells.
- **Segmentation Performance:** Demonstrated superior performance in segmenting invasive tumor and stroma regions, using Jaccard loss to optimize the segmentation process.
- **TILs Score:** The automated TILs scoring method correlated well with clinical observations, providing a valuable tool for prognostic evaluation.

## Challenges

- **Gigapixel Image Processing:** Handling whole-slide images (gigapixel in size) is computationally intensive and required effective patching and distributed processing.
- **Staining Variations:** Color normalization was crucial to mitigate staining differences that could hinder model performance.
- **Small Object Detection:** Detecting small immune cells required precise detection models like YOLOv8, which were fine-tuned for multi-scale detection.

## Contributing

We welcome contributions to improve the models and methods used in this project. If you have ideas or enhancements, feel free to submit a pull request!
