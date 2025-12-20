#!/bin/bash
# File: folders
# Author: Denis Kurka
# Year: 2025
# License: CC0


conda activate vitis-ai-tensorflow

# folders
export BUILD=./build
export TARGET_TEMPLATE=./target_template
export TARGET=${BUILD}/target
export LOG=${BUILD}/logs
export TB_LOG=${BUILD}/tb_logs
export KERAS=./
export FREEZE=${BUILD}/freeze
export COMPILE=${BUILD}/compile/
export QUANT=${BUILD}/quantize
export TFCKPT_DIR=${BUILD}/tf_chkpt

# make the necessary folders
mkdir -p ${KERAS}
mkdir -p ${LOG}

# logs & results files
export TRAIN_LOG=train.log
export KERAS_LOG=keras_2_tf.log
export FREEZE_LOG=freeze.log
export EVAL_FR_LOG=eval_frozen_graph.log
export QUANT_LOG=quant.log
export EVAL_Q_LOG=eval_quant_graph.log
export COMP_LOG=compile.log

# Keras checkpoint file
export K_MODEL=ellipse_regressor.h5

# TensorFlow files
export FROZEN_GRAPH=frozen_graph.pb
export TFCKPT=tf_float.ckpt

# calibration list file
export CALIB_LIST=calib_list.txt
export CALIB_IMAGES=1000

# ---------------------------------------------------------
# NETWORK PARAMETERS (CRITICAL UPDATES)
# ---------------------------------------------------------

# Ensure these match your training! (128 or 256)
export INPUT_HEIGHT=256
export INPUT_WIDTH=256

# Vitis Quantizer usually prefers a '?' or '1' for batch size.
export INPUT_SHAPE=?,${INPUT_HEIGHT},${INPUT_WIDTH},1

# Input Node: This comes from the first layer of your model
export INPUT_NODE=cnn_input

# Output Node: We renamed this to 'prediction' in keras_2_tf.py
export OUTPUT_NODE=prediction

export NET_NAME=ellipse_regressor

# ---------------------------------------------------------

# training parameters
export EPOCHS=100
export BATCHSIZE=8
export LEARNRATE=0.001

# target board
export BOARD=KV260
export ARCH=/workspace/FZ5-UNET/vitis_ai/ellipse_regressor/arch.json

# DPU mode
export DPU_MODE=normal
