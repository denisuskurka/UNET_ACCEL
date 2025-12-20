#!/usr/bin/env python
# File: vitis_ai/ellipse_regressor/train_evaluation.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
predict_regressor.py - Load a trained Ellipse Regression model and visualize predictions vs Ground Truth.

Usage:
    python predict_regressor.py -m ellipse_regressor.h5 -i ./data/images -md ./data/masks -o ./predictions_reg
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf

# 1. Force CPU usage (Vitis AI compatibility)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. Enable Eager Execution (TF 1.x compatibility)
if tf.__version__.startswith('1.'):
    tf.compat.v1.enable_eager_execution()

DIVIDER = '-----------------------------------------'

def predict_images(model_path, input_dir, mask_dir, output_dir, height, width):
    # 1. Load the model
    print(f"Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 3. Get list of images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
    print(f"Found {len(image_files)} images in {input_dir}")

    print(DIVIDER)

    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(mask_dir, filename) # Assume mask has same filename
        
        # --- Preprocessing ---
        # Read image in Grayscale
        original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print(f"Could not read {filename}, skipping.")
            continue

        # Resize to model input size
        img_resized = cv2.resize(original_img, (width, height))
        
        # Normalize to [0, 1] for inference
        img_norm = img_resized.astype('float32') / 255.0
        
        # Expand dims: (1, Height, Width, 1)
        img_input = img_norm.reshape((1, height, width, 1))

        # --- Inference ---
        # Output shape: (1, 5) -> [cx, cy, axis1, axis2, angle]
        preds = model.predict(img_input)
        params = preds[0] # Extract the vector

        # --- Visualization (Prediction) ---
        # Convert grayscale to BGR so we can draw a colored ellipse
        vis_img = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        cx, cy, ax1, ax2, angle = params
        
        # Safety checks
        if ax1 < 0: ax1 = 1
        if ax2 < 0: ax2 = 1
        
        center = (int(cx), int(cy))
        axes = (int(ax1), int(ax2))
        
        # Draw Predicted Ellipse in GREEN
        cv2.ellipse(vis_img, center, axes, float(angle), 0, 360, (0, 255, 0), 2)
        # Draw Center in RED
        cv2.circle(vis_img, center, 3, (0, 0, 255), -1)
        
        # Add Label text
        cv2.putText(vis_img, "Prediction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # --- Visualization (Ground Truth Mask) ---
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask_img, (width, height))
            
            # Convert mask to BGR so we can stack it with the prediction image
            mask_vis = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            
            # Optional: Draw the GT ellipse if you want, or just keep the white mask
            # For now, we leave it as the raw mask for comparison
            cv2.putText(mask_vis, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Create a black placeholder if mask is missing
            mask_vis = np.zeros_like(vis_img)
            cv2.putText(mask_vis, "No Mask Found", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Concatenate Side-by-Side ---
        # [ Prediction ] | [ Mask ]
        combined_result = np.hstack((vis_img, mask_vis))

        # --- Saving ---
        save_path = os.path.join(output_dir, f"pred_{filename}")
        cv2.imwrite(save_path, combined_result)
        
        # Progress Bar
        sys.stdout.write(f"\rProcessed {i+1}/{len(image_files)}: {filename}")
        sys.stdout.flush()

    print(f"\n{DIVIDER}")
    print(f"Done! Predictions saved to '{output_dir}'")

def run_main():
    parser = argparse.ArgumentParser(description="Run predictions with Ellipse Regressor")
    
    parser.add_argument('-m', '--model', type=str, default='ellipse_regressor.h5',
                        help='Path to the .h5 model file')
    parser.add_argument('-i', '--input_dir', type=str, default='./data/images',
                        help='Directory containing input images')
    parser.add_argument('-md', '--mask_dir', type=str, default='./data/masks',
                        help='Directory containing ground truth masks')
    parser.add_argument('-o', '--output_dir', type=str, default='./predictions_reg',
                        help='Directory to save predictions')
    parser.add_argument('-ih', '--input_height', type=int, default=128,
                        help='Model input height')
    parser.add_argument('-iw', '--input_width', type=int, default=128,
                        help='Model input width')

    args = parser.parse_args()

    predict_images(
        args.model, 
        args.input_dir,
        args.mask_dir,
        args.output_dir, 
        args.input_height, 
        args.input_width
    )

if __name__ == "__main__":
    run_main()
