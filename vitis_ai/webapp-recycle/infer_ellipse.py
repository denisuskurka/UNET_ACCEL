#!/usr/bin/env python
"""
Inference script for Ellipse U-Net (Segmentation).
Extracts ellipse parameters from the predicted mask using OpenCV.
Compatible with Vitis AI TensorFlow 1.x environments.
"""

import os

# 1. Force CPU usage (Universal fix)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Vitis AI imports
import xir
import vart

# 2. Enable Eager Execution for TF 1.x compatibility
if tf.__version__.startswith('1.'):
    tf.compat.v1.enable_eager_execution()

# ----------------------------
# Parameters
# ----------------------------
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Note: Filenames suggest segmentation (U-Net)
SW_MODEL_PATH = "ellipse_runet256.h5"
HW_MODEL_PATH = "ellipse_runet256.xmodel"

INPUT_PIC = "./data/data_cropped_final.png"
OUTPUT_EXPORT = "./data/ellipse_infer.png"

# ----------------------------
# Globals
# ----------------------------
ellipse_model_sw = None
dpu_runner = None
g_graph = None
dpu_input_ndim = None
dpu_output_ndim = None


def fit_ellipse_cv2(mask_prob):
    """
    Takes a probability mask (H, W), thresholds it, and fits an ellipse using OpenCV.
    Returns params tuple: (cx, cy, axis1, axis2, angle)
    """
    # 1. Threshold probabilities to create binary mask (0 or 255)
    # mask_prob is typically float 0.0 to 1.0
    mask_uint8 = (mask_prob > 0.5).astype(np.uint8) * 255

    # 2. Find Contours
    # cv2.findContours returns (contours, hierarchy) in OpenCV 4.x
    # or (img, contours, hierarchy) in OpenCV 3.x. This handles both.
    cnts_info = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    if not contours:
        print("[WARN] No contour found in mask. Returning default ellipse.")
        return (64, 64, 10, 10, 0) # Default center

    # 3. Get largest contour (main object)
    c = max(contours, key=cv2.contourArea)

    # 4. Fit Ellipse (requires at least 5 points)
    if len(c) < 5:
        print("[WARN] Contour too small to fit ellipse.")
        return (64, 64, 10, 10, 0)

    # box is ((cx, cy), (width, height), angle)
    ((cx, cy), (w, h), angle) = cv2.fitEllipse(c)
    
    # We return axes as semi-axes (radius), so divide width/height by 2
    return cx, cy, w/2, h/2, angle


def draw_ellipse_on_image(image_np, params, color=(0, 255, 0)):
    """
    Draws the ellipse on the image.
    """
    if image_np.dtype == np.float32:
        img_uint8 = (image_np * 255).astype(np.uint8)
    else:
        img_uint8 = image_np.astype(np.uint8)
        
    if len(img_uint8.shape) == 2:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_uint8

    cx, cy, axis1, axis2, angle = params
    center = (int(cx), int(cy))
    axes = (int(axis1), int(axis2))
    angle = float(angle)

    cv2.ellipse(img_bgr, center, axes, angle, 0, 360, color, 2)
    return img_bgr


def load_and_preprocess_image_tf(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    original_shape = tf.shape(image)[:2]
    image_resized = tf.image.resize(image, [height, width])
    return image, image_resized, original_shape


def preprocess_image_dpu(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_shape = img_gray.shape
    img_resized = cv2.resize(img_gray, (width, height), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    input_data = img_norm.reshape((1, height, width, 1))
    
    return input_data, original_shape, img_gray


def load_ellipse_model(hw=False):
    global ellipse_model_sw, dpu_runner, g_graph, dpu_input_ndim, dpu_output_ndim

    if hw:
        if dpu_runner is not None: return

        print(f"[INFO] Loading HW Model: {HW_MODEL_PATH}")
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
        print(f"[INFO] DPU Model Loaded.")
    else:
        if ellipse_model_sw is not None: return
        print(f"[INFO] Loading SW Model: {SW_MODEL_PATH}")
        ellipse_model_sw = tf.keras.models.load_model(SW_MODEL_PATH, compile=False)


def infer_ellipse(hw=False):
    load_ellipse_model(hw=hw)

    raw_mask = None
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
        
        # Output is likely (1, 128, 128, 1) or (1, 128, 128)
        raw_mask = np.squeeze(outputData[0][0])
        
    else:
        print("[INFO] Running Inference on CPU (TF)...")
        original_image, image_resized, original_shape = load_and_preprocess_image_tf(INPUT_PIC)
        image_batch = tf.expand_dims(image_resized, axis=0)
        
        pred = ellipse_model_sw.predict(image_batch)
        # Pred shape is (1, 128, 128, 1) -> Squeeze to (128, 128)
        raw_mask = np.squeeze(pred)
        
        image_to_draw_on = image_resized.numpy().squeeze()
        orig_h, orig_w = int(original_shape[0]), int(original_shape[1])

    # -----------------------------
    # POST-PROCESSING
    # -----------------------------
    
    # CRITICAL FIX: Extract params from the mask using OpenCV
    print(f"[INFO] Mask shape: {raw_mask.shape}. Fitting ellipse...")
    ellipse_params = fit_ellipse_cv2(raw_mask)
    
    print("Calculated ellipse params:", ellipse_params)

    # Draw ellipse
    image_with_ellipse = draw_ellipse_on_image(image_to_draw_on, ellipse_params)

    # Upscale
    output_upscaled = cv2.resize(
        image_with_ellipse,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(OUTPUT_EXPORT, output_upscaled)
    print(f"Saved output image to '{OUTPUT_EXPORT}'.")

    return ellipse_params

if __name__ == "__main__":
    infer_ellipse(hw=False)
