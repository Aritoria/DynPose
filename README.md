# DynPose: Dynamic Human Pose Estimation Framework

## Overview

Top-down approaches for human pose estimation (HPE) have reached a high level of sophistication, exemplified by models such as HRNet and ViTPose. However, the low efficiency of top-down methods remains a recognized issue that has not been sufficiently explored in current research. 

Our analysis reveals that the primary cause of inefficiency stems from the substantial diversity found in pose samples. Simple poses can be accurately estimated without requiring the computational resources of larger models, while a more prominent issue arises from the abundance of bounding boxes that remain excessive even after NMS.

DynPose is a straightforward yet effective dynamic framework designed to match diverse pose samples with the most appropriate models, thereby ensuring optimal performance and high efficiency. The framework contains a lightweight router and two pre-trained HPE models (one small and one large). The router is optimized to classify samples and dynamically determine the appropriate inference paths.

## Key Features

- **Dynamic Routing**: Lightweight router intelligently directs samples to appropriate models
- **Efficiency**: Achieves ~50% speed improvement over HRNet-W32 while maintaining accuracy
- **Generalization**: Works with various pre-trained models and datasets without retraining
- **Flexibility**: Compatible with ResNet-50, HRNet-W32, and other HPE architectures

## Installation & Setup

### 1. Environment Configuration
Download and configure the required environment from the official [MMPose repository]

### 2. File Replacement
Replace the following directories in your MMPose project with those from this repository:
- `config/` directory
- `model/` directory

### 3. Model Download
Download the following pre-trained models:

**MMPose Official Pre-trained Models:**
- ResNet-50 pre-trained model
- HRNet-w32 pre-trained model

**DynPose Pre-trained Files:**
- Download this project's specific pre-trained files

## Results

Extensive experiments demonstrate the effectiveness of DynPose framework. Using ResNet-50 and HRNet-W32 as pre-trained models, our DynPose achieves:

- **~50% speed increase** over HRNet-W32
- **Same-level accuracy** maintained
- **No retraining or fine-tuning** required for generalization

## Usage

After completing the installation steps, run the test code to verify model performance.


