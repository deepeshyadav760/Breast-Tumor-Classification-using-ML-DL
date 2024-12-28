# Breast Tumor Classification using Machine Learning and Deep Learning

This project is focused on developing a robust classification model to predict the stage of breast tumors (Non-Cancer, Early Phase, Middle Phase) using advanced machine learning (ML) and deep learning (DL) techniques. It leverages state-of-the-art algorithms to aid early and accurate detection of breast cancer stages, improving treatment outcomes.

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Approaches](#modeling-approaches)
- [Results](#results)
- [Challenges Faced](#challenges-faced)
- [Future Scope](#future-scope)
- [Prediction Flow](#prediction-flow)
- [Key Findings](#key-findings)
- [License](#license)

## Introduction

Breast cancer remains one of the most challenging health issues worldwide. This project aims to provide a reliable tool for clinicians by classifying breast tumor images into distinct stages, leveraging the power of both ML and DL algorithms.

## Problem Statement

Develop a system capable of:

- Accurate detection and classification of breast tumor stages using imaging data.
- Overcoming challenges such as variability in tumor size, texture, and clarity.

Techniques used include:

- Image Processing: Enhancement, segmentation (Otsu's thresholding), feature extraction (HOG), and Canny Edge Detection.
- ML Models: Random Forest, SVM.
- DL Models: Custom CNN and MobileNetV2.

## Dataset Overview

The dataset contains approximately 20,403 images classified into three categories:

- Non-Cancer: 8,060 images
- Early Phase: 6,133 images
- Middle Phase: 6,210 images

## Data Preprocessing

- **Image Resizing**: Standardized to 128x128 dimensions.
- **Normalization**: Scaled pixel values to a 0-1 range.
- **Grayscale Conversion**: Simplified computations by converting images to grayscale.
- **Augmentation**: Horizontal flip, random rotation, contrast normalization, etc.

## Modeling Approaches

### Machine Learning
1. **Support Vector Machine (SVM)**:
   - Achieved an accuracy of 74% using HOG features.
2. **Random Forest (RF)**:
   - Outperformed SVM with 77% accuracy due to better handling of high-dimensional data.

### Deep Learning
1. **Custom CNN**:
   - Achieved 80.6% accuracy on the test dataset, showing superior feature learning capabilities.
2. **MobileNetV2**:
   - Lightweight and efficient but underperformed with the dataset due to its reliance on pre-trained weights.

## Results

- The **Custom CNN** emerged as the best model with 80.6% accuracy.
- Feature learning and architecture flexibility made CNNs more effective for this domain.

## Challenges Faced

- **Data Imbalance**: Limited and imbalanced dataset impacted generalization.
- **Computational Demands**: CNNs required significant resources.
- **Transfer Learning Limitations**: Pre-trained models like MobileNetV2 were less effective for domain-specific tasks.

## Future Scope

- **Hybrid Models**: Combining CNNs with traditional ML models for better performance.
- **Real-time Detection**: Deploying on edge devices for real-time analysis.
- **Global Adaptability**: Supporting multiple languages and resource-constrained environments.
- **Enhanced Features**: Integrating multi-modal imaging and genetic data for improved predictions.

## Prediction Flow

1. Input image is uploaded for analysis.
2. Preprocessing and feature extraction steps are applied.
3. ML or DL model predicts the tumor stage.
4. Output includes stage classification and confidence scores.

## Key Findings

- Custom CNNs excelled in preserving high-resolution details necessary for stage classification.
- Training from scratch allowed domain-specific pattern learning, outperforming pre-trained models like MobileNetV2.

## Clone the Repository

To get started with this project, clone the repository:

```bash
git clone https://github.com/deepeshyadav760/Breast-Tumor-Classification-using-ML-DL
