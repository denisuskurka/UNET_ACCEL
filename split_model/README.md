
# UNet Brevitas Segmentation Project

A PyTorch-based project for image segmentation using a quantized U-Net model (UNetBrevitas) with Brevitas. This repository includes scripts for data preparation, model training, inference, deployment, and performance benchmarking, along with mechanisms for logging and visual inspection of training progress.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Directory Structure](#directory-structure)
4. [Data Preparation](#data-preparation)
5. [Training the Model](#training-the-model)
6. [Inference](#inference)
7. [Deployment to FZ5](#deployment-to-fz5)
8. [Testbench and Performance Evaluation](#testbench-and-performance-evaluation)
9. [Logging and Visualization](#logging-and-visualization)
10. [Additional Notes](#additional-notes)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## Project Overview

This project aims to perform image segmentation using a quantized U-Net architecture (`UNetBrevitas`) with the Brevitas library for quantization-aware training. The pipeline includes:

- **Data Preparation**: Processing raw images and masks, including cropping and color-based mask extraction.
- **Model Training**: Training the U-Net model with combined BCE (Binary Cross Entropy) and Dice loss to handle class imbalance.
- **Inference**: Running the trained model on test images to generate segmentation masks.
- **Deployment**: Automating deployment to FZ5 hardware for edge inference.
- **Performance Evaluation**: Benchmarking inference performance, including timing and memory usage.
- **Logging & Visualization**: Recording training progress and visualizing model predictions during training.

---

## Requirements

Ensure you have the following installed before running the scripts:

- **Python**: 3.7 or higher
- **PyTorch**: Compatible with your CUDA version (if using GPU)
- **Brevitas**: For quantization
- **OpenCV**: For image processing
- **Matplotlib**: For visualization

### Installation

It's recommended to use a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install torch torchvision brevitas opencv-python matplotlib numpy
```

---

## Directory Structure

Here’s the updated directory structure for the project:

```
/model_construct
│
├── __pycache__             # Cached Python files
├── data                    # Raw and processed data
├── deployment              # Temporary folder for deployment files
├── fz5                     # Scripts specific to FZ5 hardware
├── SN                      # Additional project files (uncategorized)
├── test_data               # Test images for inference and testing
├── venv                    # Python virtual environment
│
├── best_unet_weights.pth   # Trained model weights
├── common.py               # Common utilities for the project
├── dataprep.py             # Data preparation script
├── dataset.py              # Custom Dataset class
├── deploy_fz5.sh           # Deployment script for FZ5 hardware
├── export_finn.py          # Export script for compatibility with FINN
├── fz5_input.py            # Input handling for FZ5
├── fz5_output.py           # Output handling for FZ5
├── infer.py                # Inference script for a single image
├── inference_folder.py     # Inference script for batch processing
├── model_bu.py             # Backup model script
├── model_quant.py          # Quantized model implementation
├── model.py                # UNetBrevitas model definition
├── README.md               # This README file
├── requirements_inference.txt # Dependencies for inference
├── requirements.txt        # General dependencies
├── run_and_copy.sh         # Shell script for testing and copying results
├── testbench.py            # Testbench for performance evaluation
├── train.py                # Training script
```

> **Note**: Adjust paths in scripts as necessary to match your system's directory structure.

---

## Data Preparation

The `dataprep.py` script processes raw images and masks, including cropping, color-based mask extraction, converting image formats, and ensuring alignment between images and masks.

### Steps Performed

1. **Color Mask Extraction**: Extracts binary masks based on specific colors (e.g., green).
2. **Image Conversion**: Converts images from JPG to PNG format and crops them.
3. **Image Duplication**: Replicates base images to match mask filenames for proper pairing.

### How to Use

1. **Configure Parameters**:

   - **Crop Margin**: Number of pixels to crop from each side of the image.
   - **Color Tolerance**: Tolerance for color extraction in mask generation.

   These can be adjusted in the script:

   ```python
   CROP_MARGIN = 110
   TOLERANCE = 70
   ```

2. **Run the Script**:

   Execute the data preparation script:

   ```bash
   python dataprep.py
   ```

   This will:

   - Process and save green masks to the `data/masks` directory.
   - Convert and save images from JPG to PNG in the `data/images` directory.
   - Duplicate images to ensure each mask has a corresponding image.

3. **Customization**:

   - **Color Masks**: Uncomment and configure sections for other colors (e.g., blue, red) as needed.
   - **Folders**: Ensure that `folder_path_green`, `folder_path_blue`, and `folder_path_red` point to the correct source directories.

---

## Training the Model

The `train.py` script handles model training, including data augmentation, loss computation, early stopping, and logging.

### Features

- **Data Augmentation**: Random flips, rotations, and brightness adjustments to enhance training data.
- **Loss Functions**: Combines BCEWithLogitsLoss and DiceLoss to handle class imbalance.
- **Early Stopping**: Stops training if validation loss does not improve for a set number of epochs.
- **Logging**: Saves training and validation losses to a CSV file and generates loss plots.
- **Visual Inspection**: Saves images showing input, ground truth, and model predictions after each epoch.

### How to Use

1. **Configure Hyperparameters**:

   Adjust settings at the top of `train.py` as needed.

2. **Run the Training Script**:

   ```bash
   python train.py
   ```

---

## Inference

The `infer.py` script performs segmentation on a single input image.

### How to Use

```bash
python3 infer.py <image_path>
```

---

## Deployment to FZ5

Run the `deploy_fz5.sh` script to automate deployment of models and files.

---

## Testbench

Run `testbench.py` to evaluate batch inference performance:

```bash
python3 testbench.py
```

---

## Logging and Visualization

- **Logs**: Training and validation losses.
- **Visualization**: Side-by-side input, ground truth, and predictions.

---

## Additional Notes

- Extensive data augmentation and combined loss mitigate imbalance.
- Model checkpointing ensures the best model is saved.