# Reduced UNET on FZ5 FPGA Board

This repository contains the complete pipeline for deploying a reduced UNET model on the FZ5 FPGA board, covering everything from model design and training to optimizations and hardware deployment.

## 1. Introduction

This repository includes numerous files, but **DO NOT WORRY**—you only need a few key scripts to make everything work. The rest are for experimentation, validation, and testing.

### Key Files:
- **Model Preparation:** `model.py`, `loss.py`, `dataset.py`
- **Training:** `train.py`
- **Synthesis:** `synth.py`
- **Evaluation:** `evaluate.sh`
- **Results Visualization:** `show_raw_results.py`

#### File Descriptions:
- **`dataset.py`**: Pairs image-mask datasets located in `./data/masks` and `./data/images`, applies augmentations, and prepares data for training.
- **`model.py`**: Defines the UNET model based on [this paper](https://thesai.org/Downloads/Volume15No1/Paper_61-FPGA_based_Implementation_of_a_Resource_Efficient_UNET_Model.pdf). Some modifications were made to fit the FZ5 FPGA's constraints.
- **`loss.py`**: Implements a custom loss function that combines DICE and Binary Cross-Entropy (BCE) for better segmentation performance.
- **`train.py`**: Standard training script with pruning. **Note:** No learning rate (LR) adjustment is used, as it negatively impacts pruning with small datasets.
- **`synth.py`**: Converts the trained model into FPGA-compatible hardware using HLS4ML. It follows these steps:
  1. Creates an HLS project
  2. Optimizes FIFO depths (critical for fitting the model into hardware)
  3. Synthesizes the hardware

### Important Notes:
- **FIFO optimization is crucial!** Without it, the model won't fit into the FPGA.
- **Ensure `IOType` is set to `io_stream`**, or the synthesis process will fail.
- **Vivado 2020.1 is required** and must be sourced before running the script:
  ```sh
  source /path/to/vivado/AMD/Vivado/2020.1/settings64.sh
  ```
- **C/RTL simulation requires an old system environment.** The recommended setup is **Ubuntu 18.04.1** in a virtual machine.
- **Petalinux is required** for generating the hardware image.

## 2. Requirements

This project depends on several frameworks and tools:
- **HLS4ML** (for FPGA synthesis)
- **Petalinux** (for system image generation)
- **Vivado 2020.1** (for hardware synthesis and verification)

### Installing HLS4ML:
Use the included scripts to create the environment:
```sh
source ./../create.sh   # Creates a new environment and installs dependencies
source ./../start.sh    # Activates the environment
```
**Note:** Some dependencies (e.g., PyTorch) must be installed manually using:
```sh
pip install <package-name>
```

## 3. Running an Already Booted System

If you already have an FZ5 board running Petalinux with the accelerator, you can directly run the inference pipeline:

### Prerequisites:
1. **FZ5 is booted** with Petalinux and the accelerator.
2. **Prepared binary image** (`./X_test.bin`).
3. **Patience** (this is still a work in progress).

### Running the Inference Pipeline:
Execute the following script:
```sh
./send_data.sh
```
This script will:
1. Copy the input data and DMA commands to the FZ5 board.
2. Run the DMA transfer and wait for processing.
3. Retrieve the processed output data.

After execution, visualize the results using:
```sh
python show_raw_results.py
```
**Note:** The output may look incorrect since the accelerator and DMA transfers are still a work in progress. If you want to verify the UNET's output independently, run:
```sh
python inference.py
```

---
This project is actively being improved. Contributions and feedback are welcome!

## 4. Projects
Lastly the folder 'quantized_pruned_cnn' contains the whole HLS and Vivado project.

The quantized_cnn_model_final_128_20.h5 file are the weight for the NN.

The xilinx_com_hls_myproject_axi_1_0.zip contains the IP.

And lastly the unet128.zip is the HW project for FZ5.