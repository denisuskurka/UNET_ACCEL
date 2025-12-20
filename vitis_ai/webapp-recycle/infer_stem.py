#!/usr/bin/env python
# File: vitis_ai/webapp-recycle/infer_stem.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import time
import os
import sys

# --- Vitis AI imports ---
# Import these globally as they are the primary target for the board
import xir
import vart
import numpy as np
import cv2

# --- CONFIGURATION ---
IMAGE_HEIGHT = 256
IMAGE_WIDTH  = 256

SW_MODEL_PATH = "stem_runet256.h5"
HW_MODEL_PATH = "stem_runet256.xmodel"

# --- GLOBALS ---
stem_model_sw = None    # For TensorFlow (SW)
dpu_runner = None       # For Vitis AI (HW)
g_graph = None          # XIR Graph
dpu_input_ndim = None
dpu_output_ndim = None

# Global placeholder for the tf module so we can use it elsewhere if loaded
tf = None

def load_stem_model(hw=False):
    """
    Loads the appropriate model (SW or HW) into global variables.
    """
    global stem_model_sw, dpu_runner, g_graph, dpu_input_ndim, dpu_output_ndim, tf

    if hw:
        # --- HW MODE (Vitis AI) ---
        if dpu_runner is not None:
            return 

        print(f"[INFO] Loading HW Model: {HW_MODEL_PATH}")

        if not os.path.exists(HW_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {HW_MODEL_PATH}")

        # XIR will use system libraries correctly since TF is not loaded yet
        g_graph = xir.Graph.deserialize(HW_MODEL_PATH)

        print("[INFO] Creating DPU Runner...")
        
        root_subgraph = g_graph.get_root_subgraph()
        child_subgraphs = root_subgraph.toposort_child_subgraph()
        dpu_subgraph = [cs for cs in child_subgraphs 
                        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"][0]

        dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
        
        inputTensors = dpu_runner.get_input_tensors()
        outputTensors = dpu_runner.get_output_tensors()
        
        dpu_input_ndim = tuple(inputTensors[0].dims)
        dpu_output_ndim = tuple(outputTensors[0].dims)

        print(f"[INFO] DPU Model Loaded. Input: {dpu_input_ndim}, Output: {dpu_output_ndim}")

    else:
        # --- SW MODE (TensorFlow) ---
        if stem_model_sw is not None:
            return 

        print("[INFO] HW=False detected. Importing TensorFlow for SW mode...")
        
        # 1. Configuration (Must be before import)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # 2. Lazy Import
        import tensorflow as _tf
        tf = _tf 

        if tf.__version__.startswith('1.'):
            tf.compat.v1.enable_eager_execution()
            print(f"[INFO] TensorFlow 1.x detected. Eager execution enabled.")

        # --- COMPATIBILITY PATCH ---
        # Define wrappers to swallow the 'dtype' argument that causes crashes
        # on older TF versions (Vitis AI images usually use TF 2.8/2.9)
        
        class PatchedVarianceScaling(tf.keras.initializers.VarianceScaling):
            def __init__(self, **kwargs):
                # Newer TF adds 'dtype', older TF crashes on it. Remove it.
                if 'dtype' in kwargs: kwargs.pop('dtype')
                super().__init__(**kwargs)

        class PatchedZeros(tf.keras.initializers.Zeros):
            def __init__(self, **kwargs):
                # Zeros in older TF takes NO arguments
                super().__init__()

        # Map the problematic class names to our patched versions
        custom_objects = {
            'VarianceScaling': PatchedVarianceScaling,
            'Zeros': PatchedZeros
        }
        # ---------------------------

        print(f"[INFO] Loading SW Model: {SW_MODEL_PATH}")
        
        # Pass custom_objects to load_model
        stem_model_sw = tf.keras.models.load_model(
            SW_MODEL_PATH, 
            compile=False,
            custom_objects=custom_objects
        )
        print("[INFO] SW Model loaded.")

def load_and_preprocess_image_tf(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    # Ensure TF is loaded (it should be if we reached here via infer_stem(hw=False))
    if tf is None:
        raise ImportError("TensorFlow not loaded. Call load_stem_model(hw=False) first.")

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])
    return image

def preprocess_image_dpu(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    # Standard OpenCV preprocessing (No TF dependency)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.reshape((1, height, width, 1))
    return img

def infer_stem(cropped_filepath, hw=False):
    if cropped_filepath is None or not os.path.exists(cropped_filepath):
        print(f"[ERROR] Image not found: {cropped_filepath}")
        return None

    # Load logic
    load_stem_model(hw=hw)

    if hw:
        # HW Inference (Uses VART, no TF)
        input_data = preprocess_image_dpu(cropped_filepath)
        inputData = [np.empty(dpu_input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(dpu_output_ndim, dtype=np.float32, order="C")]
        inputData[0][0, ...] = input_data[0]
        
        job_id = dpu_runner.execute_async(inputData, outputData)
        dpu_runner.wait(job_id)
        
        pred_mask = outputData[0][0]
        pred_mask = np.squeeze(pred_mask)
        return pred_mask

    else:
        # SW Inference (Uses TF)
        image = load_and_preprocess_image_tf(cropped_filepath)
        image_batch = tf.expand_dims(image, axis=0)
        pred = stem_model_sw.predict(image_batch)
        pred_mask = np.squeeze(pred)
        return pred_mask
