#!/usr/bin/env python
import time
import os

# 1. Force TensorFlow to use the CPU only (Works in both TF1 and TF2)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import cv2

# Vitis AI imports
import xir
import vart

# 2. Enable Eager Execution for TF 1.x compatibility
# This ensures .numpy() and other TF2-style commands work in the Vitis AI environment
if tf.__version__.startswith('1.'):
    tf.compat.v1.enable_eager_execution()
    print(f"[INFO] TensorFlow 1.x detected. Eager execution enabled.")

# --- CONFIGURATION ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH  = 128

SW_MODEL_PATH = "stem_unet128.h5"
HW_MODEL_PATH = "stem_unet128.xmodel"

# --- GLOBALS ---
stem_model_sw = None    # For TensorFlow (SW)
dpu_runner = None       # For Vitis AI (HW)
g_graph = None          # XIR Graph
dpu_input_ndim = None
dpu_output_ndim = None

def load_stem_model(hw=False):
    """
    Loads the appropriate model (SW or HW) into global variables.
    """
    global stem_model_sw, dpu_runner, g_graph, dpu_input_ndim, dpu_output_ndim

    if hw:
        if dpu_runner is not None:
            return # Already loaded

        print(f"[INFO] Loading HW Model: {HW_MODEL_PATH}")
        # Deserialize the xmodel
        g_graph = xir.Graph.deserialize(HW_MODEL_PATH)
        
        # Find the DPU subgraph
        root_subgraph = g_graph.get_root_subgraph()
        child_subgraphs = root_subgraph.toposort_child_subgraph()
        dpu_subgraph = [cs for cs in child_subgraphs 
                        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"][0]

        # Create the DPU runner
        dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")

        # Get tensor shapes for I/O
        inputTensors = dpu_runner.get_input_tensors()
        outputTensors = dpu_runner.get_output_tensors()
        
        dpu_input_ndim = tuple(inputTensors[0].dims)   # e.g., (1, 128, 128, 1)
        dpu_output_ndim = tuple(outputTensors[0].dims) # e.g., (1, 128, 128, 1)

        print(f"[INFO] DPU Model Loaded. Input: {dpu_input_ndim}, Output: {dpu_output_ndim}")

    else:
        if stem_model_sw is not None:
            return # Already loaded

        print(f"[INFO] Loading SW Model: {SW_MODEL_PATH}")
        # compile=False allows loading without needing custom optimizers/losses
        stem_model_sw = tf.keras.models.load_model(
            SW_MODEL_PATH, compile=False
        )
        print("[INFO] SW Model loaded.")

def load_and_preprocess_image_tf(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    TF-based preprocessing for SW mode.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])
    return image

def preprocess_image_dpu(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    OpenCV/Numpy-based preprocessing for HW/DPU mode.
    Returns a numpy array with shape (1, H, W, 1) and dtype float32.
    """
    # Read grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to 0-1 (Float32)
    img = img.astype(np.float32) / 255.0
    
    # Reshape to (1, H, W, 1)
    img = img.reshape((1, height, width, 1))
    
    return img

def infer_stem(cropped_filepath, hw=False):
    """
    Runs inference using either TF (CPU) or Vitis AI (DPU).
    """
    if cropped_filepath is None or not os.path.exists(cropped_filepath):
        print(f"[ERROR] Image not found: {cropped_filepath}")
        return None

    # Ensure the correct model is loaded
    load_stem_model(hw=hw)

    if hw:
        # --- HW / DPU INFERENCE ---
        # 1. Preprocess using OpenCV (avoid TF graph overhead for HW run)
        input_data = preprocess_image_dpu(cropped_filepath)
        
        # 2. Prepare I/O buffers (Must be C-Contiguous for VART)
        # We assume batch size 1 for single image inference
        inputData = [np.empty(dpu_input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(dpu_output_ndim, dtype=np.float32, order="C")]
        
        # 3. Copy data into input buffer
        inputData[0][0, ...] = input_data[0]
        
        # 4. Execute Async on DPU
        job_id = dpu_runner.execute_async(inputData, outputData)
        dpu_runner.wait(job_id)
        
        # 5. Extract result
        # outputData[0] is (Batch, H, W, C).
        pred_mask = outputData[0][0]
        
        # Squeeze to (H, W) or (H, W, 1) depending on caller expectation
        pred_mask = np.squeeze(pred_mask)
        
        return pred_mask

    else:
        # --- SW / TENSORFLOW INFERENCE ---
        # 1. Preprocess
        image = load_and_preprocess_image_tf(cropped_filepath)
        image_batch = tf.expand_dims(image, axis=0) # (1, H, W, 1)
        
        # 2. Predict
        # print("[INFO] Running on CPU (TF)...")
        pred = stem_model_sw.predict(image_batch)
        pred_mask = np.squeeze(pred)

        return pred_mask
