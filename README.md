# U-Net TIFF Image Segmentation Project

## Project Title  
U-Net Image Segmentation using TIFF Data

---

## 🔍 Overview

This project trains a **U-Net deep learning model** to perform pixel-wise segmentation on grayscale TIFF image stacks. It is designed to run in **Google Colab** with data stored in Google Drive.

The complete pipeline:

- Loads TIFF image and mask stacks (`train_data.tif`, `train_label.tif`)
- Splits large images into 256×256 patches
- Normalizes images and binarizes masks
- Performs a 75/25 train–test split
- Applies real-time data augmentation
- Builds a custom U-Net convolutional neural network
- Trains the model on the dataset
- Plots training and validation performance
- Saves the trained model as an `.h5` file

The goal is accurate binary segmentation of microscopy images (e.g., mitochondria detection).

---

## Model Architecture

The network follows the original U-Net architecture:

**U-Net: Convolutional Networks for Biomedical Image Segmentation**  
Ronneberger et al., 2015

### Key Components

- Encoder–decoder structure  
- Skip connections between contracting and expanding paths  
- Convolution → BatchNorm → ReLU blocks  
- Bottleneck layer (1024 filters)  
- Final 1×1 convolution with sigmoid activation  

### Training Configuration

- Optimizer: Adam (learning rate = 1e-3)  
- Loss: Binary Crossentropy  
- Metric: Accuracy  
- Batch size: 8  
- Epochs: 25  

---

## Data Processing & Augmentation

### Preprocessing

- Images are normalized to `[0, 1]`
- Masks are converted to binary
- Patches are reshaped to include a channel dimension

### Data Augmentation

Uses `ImageDataGenerator` with:

- Rotation  
- Width and height shifts  
- Shear  
- Zoom  
- Horizontal and vertical flips  
- Reflection padding  

Masks are re-binarized after augmentation to prevent interpolation artifacts.

---

## 📁 Required Files

Place the following files in your working directory:
train_data.tif
train_label.tif

---

## ️ Installation

1. Make sure Python 3 is installed.
2. Install the required libraries:

```bash
pip install -r requirements.txt
