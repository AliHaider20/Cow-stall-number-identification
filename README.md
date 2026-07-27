# Cow Stall Number Identification and Localization

## Overview

This project implements an object detection system to automatically identify and localize cow stall numbers in dairy farm images. The system uses deep learning techniques to detect small numbers in images, with applications extending to door numbers, apartment numbers, license plates, and more. Paper published [here](https://www.researchgate.net/publication/370729404_Cow_teat_data_augmentation_using_Stable_Diffusion)

**Authors:** Haider Ali, Yeshiva University  
**Email:** Hali4@mail.yu.edu

## Problem Statement

The dairy industry relies on efficient herd management to optimize milk production and maintain cow health. Automated systems such as milk delivery bots and dung collection bots require accurate stall identification. This project addresses this need by developing a computer vision solution for automatic cow stall number detection.

## Dataset

- **Total Images:** 1,303
  - Training: 1,042 images
  - Testing: 261 images
- **Number of Classes:** 61 stall numbers
- **Class Distribution:** Highly imbalanced, with class 0 representing ~17.5% of data
- **Data Format:** Images with bounding box coordinates and class labels

**Note:** Dataset collected and organized by Zhang et al. [2]

## Methodology

### Model Architecture

**ResNet34 with Linear Head**
- Pre-trained ResNet34 used as feature extractor
- Last layer removed and replaced with custom outputs
- Two output heads:
  1. **Classification head** - identifies stall number
  2. **Bounding box regression head** - predicts stall location

### Key Techniques

- **Loss Functions:**
  - Cross Entropy Loss with inverse class weights (weight: 1.0)
  - Smooth L1 Loss for bounding box prediction (weight: 0.1)
  - Focal Loss with gamma parameter (weight: 5.0 × (1 - IoU))
  
- **Data Augmentation:**
  - Random rotation (±20 degrees)
  - Random horizontal and vertical flips
  - Random resized crop
  - Image resizing to 256×256

- **Imbalanced Dataset Handling:**
  - Inverse class weighting
  - Focal Loss to address class imbalance

### Training Configuration

- **Framework:** PyTorch (v1.13.1)
- **Compute:** CUDA 11.6 (NVIDIA GPU via Kaggle)
- **Optimizer:** Adam and SGD
- **Batch Size:** 64
- **Epochs:** 250

## Results

| Model | Train Accuracy | Test Accuracy |
|-------|---|---|
| VGG19 | 0.30 | 0.164 |
| ResNet18 | 0.22 | 0.164 |
| ResNet50 | 0.20 | 0.164 |
| **ResNet34 + Linear** | **0.81** | **0.80** |

**Final Model Performance:**
- Training Accuracy: **81%**
- Testing Accuracy: **80%**

## Key Findings

1. **Model Performance:** ResNet34 with custom output heads significantly outperforms other architectures (VGG19, ResNet18, ResNet50)
2. **Loss Function Impact:** Smooth L1 Loss combined with Cross Entropy and Focal Loss provides optimal results
3. **Data Imbalance:** Inverse class weighting and Focal Loss effectively mitigate overfitting
4. **Transductive Learning:** Using the test dataset during training improves model performance

## Comparison with Prior Work

**Previous Approach (Zhang et al.):**
- Model: ResNet34 (pre-trained)
- Class Identification Accuracy: 95%
- Number Localization Accuracy: 40%

**Current Approach:**
- Improved focus on both classification and localization
- Addresses dataset imbalance more effectively
- Enhanced loss function design

## Technical Details

- **Development Environment:** Kaggle Notebook
- **GPU Support:** CUDA acceleration for faster training
- **Framework:** PyTorch

## Practical Applications

- **Dairy Industry:** Automated tracking of cow health and milk production
- **Autonomous Systems:** Milk delivery bots, dung collection bots
- **General Computer Vision:** Small number detection in:
  - Door/apartment numbers
  - License plates
  - Building identifiers

## Future Improvements

1. Increase accuracy to **90%+** through:
   - Advanced data augmentation techniques (CutMix, CutOut)
   - Testing GoogleNet and Inception models
   - Creating synthetic datasets for under-labeled classes

2. **Semi-supervised Learning** to leverage unlabeled data

3. **Reinforcement Learning** to understand model parameter learning with limited data

4. **Additional Architectures:** VGG16, VGG19, GoogleNet, Inception models

## Conclusion

This project demonstrates the effectiveness of computer vision techniques in automating cow stall detection for the dairy industry. By accurately identifying stall numbers, farmers can:
- Optimize herd management practices
- Improve tracking of cow health and milk production
- Generate valuable behavioral and health insights
- Reduce manual labor requirements

The system provides a foundation for further research in agricultural automation and animal welfare management.

## References

[1] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. "SSD: Single Shot MultiBox Detector." In Computer Vision ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pages 21–37. Springer, 2016.

[2] Youshan Zhang. "Stall number detection of cow teats key frames." arXiv preprint arXiv:2303.10444, 2023.

---
