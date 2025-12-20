#!/usr/bin/env python
# File: vitis_ai/webapp-recycle/infer_ellipse_regress.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
Inference script for Ellipse Regression Model.
Directly predicts parameters (cx, cy, axis1, axis2, angle) and draws them.
Compatible with Vitis AI (SW and HW/DPU).
"""

import os
import time
import numpy as np
import cv2

# Vitis AI imports (Safe to import globally)
import xir
import vart

# ----------------------------
# Configuration
# ----------------------------
IMAGE_HEIGHT = 256
IMAGE_WIDTH  = 256

SW_MODEL_PATH = "ellipse_regressor.h5"
HW_MODEL_PATH = "ellipse_regressor.xmodel"

INPUT_PIC = "./data/data_cropped_final.png"
OUTPUT_EXPORT = "./data/ellipse_infer.png"

# ----------------------------
# Globals
# ----------------------------
regressor_model_sw = None
dpu_runner = None
g_graph = None
dpu_input_ndim = None
dpu_output_ndim = None

# Global placeholder for TensorFlow
tf = None

def load_and_preprocess_image_tf(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    TF-based preprocessing for SW inference.
    """
    if tf is None:
        raise ImportError("TensorFlow not loaded. Call load_regressor_model(hw=False) first.")

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    original_shape = tf.shape(image)[:2]
    image_resized = tf.image.resize(image, [height, width])
    return image, image_resized, original_shape

def preprocess_image_dpu(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    OpenCV-based preprocessing for HW inference.
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_shape = img_gray.shape
    
    # Resize
    img_resized = cv2.resize(img_gray, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Normalize [0,1]
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Reshape (1, H, W, 1)
    input_data = img_norm.reshape((1, height, width, 1))
    
    return input_data, original_shape, img_gray

def draw_ellipse_on_image(image_np, params, color=(0, 255, 0)):
    """
    Draws the ellipse using direct parameters [cx, cy, ax1, ax2, angle]
    """
    if image_np.dtype == np.float32:
        img_uint8 = (image_np * 255).astype(np.uint8)
    else:
        img_uint8 = image_np.astype(np.uint8)
        
    if len(img_uint8.shape) == 2:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_uint8.copy()

    cx, cy, axis1, axis2, angle = params
    
    if axis1 <= 0: axis1 = 1
    if axis2 <= 0: axis2 = 1

    center = (int(cx), int(cy))
    axes = (int(axis1), int(axis2))
    angle = float(angle)

    cv2.ellipse(img_bgr, center, axes, angle, 0, 360, color, 2)
    cv2.circle(img_bgr, center, 3, (0, 0, 255), -1)
    
    return img_bgr

def load_regressor_model(hw=False):
    global regressor_model_sw, dpu_runner, g_graph, dpu_input_ndim, dpu_output_ndim, tf

    if hw:
        # --- HW MODE (Vitis AI) ---
        if dpu_runner is not None: return

        print(f"[INFO] Loading HW Model: {HW_MODEL_PATH}")
        if not os.path.exists(HW_MODEL_PATH):
             print(f"[ERROR] xmodel file not found: {HW_MODEL_PATH}")
             return

        g_graph = xir.Graph.deserialize(HW_MODEL_PATH)
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
        if regressor_model_sw is not None: return
        
        print("[INFO] HW=False detected. Importing TensorFlow...")
        
        # 1. Force CPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        # 2. Lazy Import
        import tensorflow as _tf
        tf = _tf

        if tf.__version__.startswith('1.'):
            tf.compat.v1.enable_eager_execution()

        # --- COMPATIBILITY PATCH START ---
        # Patched classes to strip 'dtype' argument from config
        
        class PatchedVarianceScaling(tf.keras.initializers.VarianceScaling):
            def __init__(self, **kwargs):
                if 'dtype' in kwargs: kwargs.pop('dtype')
                super().__init__(**kwargs)

        class PatchedZeros(tf.keras.initializers.Zeros):
            def __init__(self, **kwargs):
                super().__init__() # Zeros takes no args in older TF

        class PatchedOnes(tf.keras.initializers.Ones):
            def __init__(self, **kwargs):
                super().__init__() # Ones takes no args in older TF

        class PatchedGlorotUniform(tf.keras.initializers.GlorotUniform):
            def __init__(self, **kwargs):
                if 'dtype' in kwargs: kwargs.pop('dtype')
                super().__init__(**kwargs)

        custom_objects = {
            'VarianceScaling': PatchedVarianceScaling,
            'Zeros': PatchedZeros,
            'Ones': PatchedOnes,
            'GlorotUniform': PatchedGlorotUniform  # Patch for Dense layers
        }
        # --- COMPATIBILITY PATCH END ---

        print(f"[INFO] Loading SW Model: {SW_MODEL_PATH}")
        
        regressor_model_sw = tf.keras.models.load_model(
            SW_MODEL_PATH, 
            compile=False,
            custom_objects=custom_objects
        )

def infer_ellipse(hw=False):
    load_regressor_model(hw=hw)

    ellipse_params = None
    image_to_draw_on = None 
    orig_h, orig_w = (0, 0)

    # -----------------------------
    # EXECUTION
    # -----------------------------
    if hw:
        print("[INFO] Running Inference on DPU...")
        input_data, (orig_h, orig_w), orig_gray = preprocess_image_dpu(INPUT_PIC)
        
        image_to_draw_on = input_data[0, ..., 0] 
        
        inputData = [np.empty(dpu_input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(dpu_output_ndim, dtype=np.float32, order="C")]
        
        inputData[0][0, ...] = input_data[0]
        
        job_id = dpu_runner.execute_async(inputData, outputData)
        dpu_runner.wait(job_id)
        
        ellipse_params = outputData[0][0]
        
    else:
        print("[INFO] Running Inference on CPU (TF)...")
        original_image, image_resized, original_shape = load_and_preprocess_image_tf(INPUT_PIC)
        image_batch = tf.expand_dims(image_resized, axis=0)
        
        preds = regressor_model_sw.predict(image_batch)
        ellipse_params = preds[0]
        
        image_to_draw_on = image_resized.numpy().squeeze()
        orig_h, orig_w = int(original_shape[0]), int(original_shape[1])

    # -----------------------------
    # POST-PROCESSING
    # -----------------------------
    print(f"Predicted Params: {ellipse_params}")

    image_with_ellipse = draw_ellipse_on_image(image_to_draw_on, ellipse_params)

    output_upscaled = cv2.resize(
        image_with_ellipse,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(OUTPUT_EXPORT, output_upscaled)
    print(f"Saved output image to '{OUTPUT_EXPORT}'.")

    return ellipse_params

if __name__ == "__main__":
    # Change to True to test DPU mode
    infer_ellipse(hw=False)
