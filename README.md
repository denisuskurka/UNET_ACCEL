# PaddlePaddle ResNet/UNet Project

This repository contains two simple training scripts using PaddlePaddle:

- **ResNet-50** training on the CIFAR-10 dataset.
- **UNet** model training on a synthetic dataset.

Additionally, it includes instructions on how to convert the saved models into mobile versions using Paddle-Lite's `opt` tool.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training ResNet-50 on CIFAR-10](#training-resnet-50-on-cifar-10)
- [Training UNet on Synthetic Dataset](#training-unet-on-synthetic-dataset)
- [Converting Models for Mobile Deployment](#converting-models-for-mobile-deployment)
- [License](#license)

## Prerequisites

- **Python** >= 3.6
- **PaddlePaddle** >= 2.0
- **Virtual Environment** (optional but recommended)
- **Requirements File**: `requirements.txt`

## Installation

### 1. Create a Virtual Environment (Recommended)

It's recommended to use a virtual environment to manage your project's dependencies without affecting your global Python environment.

#### For Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

Make sure you have `pip` installed and updated:

```bash
pip install --upgrade pip
```

Install all required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` file should include all necessary packages, including `paddlepaddle`, `numpy`, `matplotlib`, etc.

### 3. Verify PaddlePaddle Installation

To ensure PaddlePaddle is installed correctly, run:

```bash
python -c "import paddle; print(paddle.__version__)"
```

You should see the installed PaddlePaddle version printed out.

## Training ResNet-50 on CIFAR-10

The script `train_resnet_cifar10.py` trains a ResNet-50 model on the CIFAR-10 dataset.

### Steps

1. **Run the Training Script**

   ```bash
   python train_resnet_cifar10.py
   ```

   **Note**: The script is set to run for 1 epoch and process 1 batch (`max_batches = 1`) for demonstration purposes. Adjust `epochs` and `max_batches` in the script as needed.

### What the Script Does

- **Loads** the CIFAR-10 dataset using `paddle.vision.datasets.Cifar10`.
- **Defines** a ResNet-50 model adjusted for 10 classes.
- **Sets up** the loss function (`CrossEntropyLoss`) and optimizer (`Adam`).
- **Trains** the model for the specified number of epochs.
- **Evaluates** the model on the test dataset.
- **Saves** the trained model to `resnet50_cifar10/model` using `paddle.jit.save`.

## Training UNet on Synthetic Dataset

The UNet model is trained on a synthetic dataset generated on-the-fly. The code is organized into three files within the `unet` directory:

- `model.py`: Defines the UNet architecture.
- `dataloader.py`: Creates a synthetic dataset.
- `train.py`: Trains the UNet model.

### Steps

1. **Navigate to the `unet` Directory**

   ```bash
   cd unet
   ```

2. **Run the Training Script**

   ```bash
   python train.py
   ```

   **Note**: Similar to the ResNet script, `train.py` is set to run for 1 epoch and process 1 batch for demonstration purposes. Adjust `epochs` and `max_batches` as needed.

### What the Script Does

- **Generates** a synthetic dataset with random images and masks.
- **Defines** the UNet model suitable for binary segmentation.
- **Sets up** the loss function (`BCEWithLogitsLoss`) and optimizer (`Adam`).
- **Visualizes** a sample image and its corresponding mask using `matplotlib`.
- **Trains** the model.
- **Saves** the trained model to `unet_model/unet` using `paddle.jit.save`.

## Converting Models for Mobile Deployment

To deploy the trained models on mobile devices, you need to convert them using Paddle-Lite's `opt` tool.

### Steps

#### 1. Download Paddle-Lite's `opt` Tool

- Visit the [Paddle-Lite Releases](https://github.com/PaddlePaddle/Paddle-Lite/releases) page.
- Download the appropriate version of the `opt` tool for your system architecture. For Linux x86 systems, download `opt_linux_x86`.

#### 2. Make the `opt` Tool Executable

```bash
chmod +x opt_linux_x86
```

#### 3. Convert the ResNet-50 Model

Run the following command in the directory containing your model files:

```bash
./opt_linux_x86 \
  --model_file=./resnet50_cifar10/model.pdmodel \
  --param_file=./resnet50_cifar10/model.pdiparams \
  --optimize_out=./resnet50_cifar10_opt \
  --valid_targets=arm
```

#### 4. Convert the UNet Model

Run the following command:

```bash
./opt_linux_x86 \
  --model_file=./unet_model/unet.pdmodel \
  --param_file=./unet_model/unet.pdiparams \
  --optimize_out=./unet_model_opt \
  --valid_targets=arm
```

**Note**: Replace `arm` in `--valid_targets=arm` with the appropriate target architecture for your mobile device, such as `opencl`, `x86`, etc.

#### 5. Result

- The converted models will be saved in `./resnet50_cifar10_opt` and `./unet_model_opt` directories.
- These models are optimized for deployment on mobile devices using Paddle-Lite.

### Additional Information

- **Paddle-Lite Documentation**: [Paddle-Lite Official Documentation](https://paddle-lite.readthedocs.io/en/latest/).
- **Supported Platforms**: Ensure your target device is supported by checking the Paddle-Lite documentation.
