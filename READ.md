# U-Net TIFF Image Segmentation Project

## Project Title
U-Net Image Segmentation using TIFF Data

---

## What This Project Does

This project trains a U-Net deep learning model to perform image segmentation on TIFF images.

The pipeline:

- Loads TIFF image and mask stacks
- Splits large images into 256x256 patches
- Preprocesses images (normalization and reshaping)
- Builds a U-Net convolutional neural network
- Trains the model on the dataset
- Saves the trained model as an `.h5` file

The goal is to perform pixel-wise segmentation on grayscale images.

---

## How to Install

1. Make sure Python 3 is installed.
2. Install the required libraries:

```bash
pip install -r requirements.txt

