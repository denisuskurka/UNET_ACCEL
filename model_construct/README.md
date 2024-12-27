# UNet Brevitas Segmentation Project

A PyTorch-based project for image segmentation using a quantized U-Net model (UNetBrevitas) with Brevitas. This repository includes scripts for data preparation, model training, and inference, along with mechanisms for logging and visual inspection of training progress.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Directory Structure](#directory-structure)
4. [Data Preparation](#data-preparation)
5. [Training the Model](#training-the-model)
6. [Inference](#inference)
7. [Logging and Visualization](#logging-and-visualization)
8. [Additional Notes](#additional-notes)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## Project Overview

This project aims to perform image segmentation using a quantized U-Net architecture (`UNetBrevitas`) with the Brevitas library for quantization-aware training. The pipeline includes:

- **Data Preparation**: Processing raw images and masks, including cropping and color-based mask extraction.
- **Model Training**: Training the U-Net model with combined BCE (Binary Cross Entropy) and Dice loss to handle class imbalance.
- **Inference**: Running the trained model on test images to generate segmentation masks.
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

Organize your project directories as follows:

```
/project_root
│
├── data
│   ├── images            # Directory to store processed images
│   └── masks             # Directory to store processed masks
│
├── progress              # Directory to store logs and visual inspections
│
├── dataset.py            # Custom Dataset class
├── model.py              # UNetBrevitas model definition
├── dataprep.py           # Data preparation script
├── train.py              # Training script
├── inference.py          # Inference script
├── README.md             # This README file
└── requirements.txt      # (Optional) List of dependencies
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

### Example Output

After running `dataprep.py`, you should have:

- Processed masks in `/data/masks` (e.g., `01_green.png`).
- Corresponding images in `/data/images` (e.g., `01.png` and `01_green.png`).

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

   Adjust settings at the top of `train.py` as needed:

   ```python
   HEIGHT, WIDTH = 160, 160       # Image & mask size
   BATCH_SIZE = 8
   NUM_EPOCHS = 200
   LEARNING_RATE = 0.0005
   VAL_SPLIT = 0.2                # 20% of data for validation
   EARLY_STOP_PATIENCE = 8        # Stop if no improvement for 8 epochs
   SAVE_BEST_MODEL_PATH = "best_unet_brevitas_weights.pth"
   
   IMAGES_DIR = "/path/to/data/images"
   MASKS_DIR  = "/path/to/data/masks"
   ```

2. **Run the Training Script**:

   Execute the training script:

   ```bash
   python train.py
   ```

   **Training Output**:

   - **Logs**: Displays training and validation loss per epoch.
   - **CSV Log**: Saves `progress/loss_log.csv` with columns `[epoch, train_loss, val_loss]`.
   - **Visuals**: Saves `progress/<epoch>_visual.png` showing input, ground truth, and prediction.
   - **Loss Plot**: At the end of training, saves `progress/loss_plot.png` showing train and val loss curves.
   - **Best Model**: Saves the best model weights to `best_unet_brevitas_weights.pth`.

3. **Training Progress Files**:

   All logs and visual inspections are stored in the `progress/` directory:

   - `loss_log.csv`: Records epoch-wise train and validation losses.
   - `loss_plot.png`: Plots train vs. val loss over epochs.
   - `1_visual.png`, `2_visual.png`, ..., `N_visual.png`: Visual inspection images per epoch.

4. **Adjust Data Augmentation**:

   The `RealSegDataset` in `dataset.py` includes various augmentations. Modify probabilities and ranges as needed within the dataset initialization in `train.py`.

---

## Inference

The `inference.py` script loads the trained model and performs segmentation on test images, displaying input and output masks.

### How to Use

1. **Ensure the Best Model is Trained**:

   Make sure `best_unet_brevitas_weights.pth` exists in your project directory from the training phase.

2. **Prepare Test Data**:

   Place your test images in the `./test_data` directory. Ensure they are in a compatible format (e.g., PNG) and match the preprocessing steps (e.g., grayscale, resized to 160x160).

3. **Run the Inference Script**:

   Execute the inference script:

   ```bash
   python inference.py
   ```

   **Inference Output**:

   - For each image in `./test_data`:
     - Displays the input image, ground truth mask (if available), and the model's predicted mask.
     - Saves a visual comparison as `progress/<epoch>_visual.png`.

4. **Troubleshooting Device Issues**:

   If you encounter device-related errors (e.g., tensors on different devices), ensure that both the model and input tensors are on the same device (`cuda` or `cpu`). The provided `inference.py` script handles this by moving the model and tensors to the appropriate device.

---

## Logging and Visualization

### CSV Logging

- **File**: `progress/loss_log.csv`
- **Contents**: Records `epoch`, `train_loss`, and `val_loss` for each epoch.
- **Usage**: Useful for analyzing training progress and identifying trends.

### Loss Plot

- **File**: `progress/loss_plot.png`
- **Contents**: Visual graph of training and validation loss over epochs.
- **Usage**: Quickly assess model convergence and potential overfitting.

### Visual Inspection Images

- **Files**: `progress/<epoch>_visual.png` (e.g., `1_visual.png`, `2_visual.png`, ...)
- **Contents**: Side-by-side comparison of input image, ground truth mask, and model prediction.
- **Usage**: Monitor qualitative improvements in model predictions throughout training.

---

## Additional Notes

- **Class Imbalance**: Given that masks occupy only 5-8% of the image, class imbalance is significant. The combined BCE + Dice loss helps mitigate this, but further strategies like weighted loss functions or focal loss can also be considered.
  
- **Data Augmentation**: Extensive augmentations (flips, rotations, brightness adjustments) help the model generalize better with limited data.

- **Early Stopping**: Prevents overfitting by halting training when validation loss ceases to improve.

- **Model Checkpointing**: Saves the best model based on validation loss, ensuring the best-performing model is retained.
