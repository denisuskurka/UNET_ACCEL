#!/usr/bin/env python
# File: 
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
predict.py - Load a trained U-Net model and generate masks for all images in a folder.

Usage:
    python predict.py -m model.hdf5 -i ./data/images -o ./predictions
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf

# Standardizing display of progress
DIVIDER = '-----------------------------------------'

def sigmoid(x):
    """Convert raw logits to probabilities."""
    return 1 / (1 + np.exp(-x))

def predict_images(model_path, input_dir, output_dir, height, width):
    # 1. Load the model
    # compile=False is crucial here because we don't need the custom loss functions
    # (like focal_tversky_loss) just to make predictions, only for training.
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
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(image_files)} images in {input_dir}")

    print(DIVIDER)

    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        
        # --- Preprocessing ---
        # Read image in Grayscale (0 flag)
        original_img = cv2.imread(img_path, 0)
        if original_img is None:
            print(f"Could not read {filename}, skipping.")
            continue

        # Resize to model input size
        img_resized = cv2.resize(original_img, (width, height))
        
        # Normalize to [0, 1] (matching training logic)
        img_norm = img_resized.astype('float32') / 255.0
        
        # Expand dims to fit model input: (1, Height, Width, 1)
        img_input = np.expand_dims(img_norm, axis=-1) # Add channel dim
        img_input = np.expand_dims(img_input, axis=0) # Add batch dim

        # --- Inference ---
        # Result is raw logits because the model has no activation at the end
        logits = model.predict(img_input, verbose=0)
        
        # Apply Sigmoid to get probabilities [0, 1]
        probs = sigmoid(logits)
        
        # Threshold to binary [0, 1]
        binary_mask = (probs > 0.55).astype(np.uint8)
        
        # Remove batch and channel dims for saving: (1, H, W, 1) -> (H, W)
        pred_mask = binary_mask[0, ..., 0]
        
        # Scale up to 0-255 for image saving
        pred_mask_img = pred_mask * 255

        # --- Visualization / Saving ---
        # Create a side-by-side comparison: [Original Resized | Prediction]
        concat_img = np.hstack((img_resized, pred_mask_img))
        
        save_path = os.path.join(output_dir, f"pred_{filename}")
        cv2.imwrite(save_path, concat_img)
        
        # Optional: Simple progress bar
        sys.stdout.write(f"\rProcessed {i+1}/{len(image_files)}: {filename}")
        sys.stdout.flush()

    print(f"\n{DIVIDER}")
    print(f"Done! Predictions saved to '{output_dir}'")

def run_main():
    parser = argparse.ArgumentParser(description="Run predictions with a trained U-Net")
    
    parser.add_argument('-m', '--model', type=str, default='./ellipse_unet.h5',
                        help='Path to the .h5 model file')
    parser.add_argument('-i', '--input_dir', type=str, default='./data/images',
                        help='Directory containing input images')
    parser.add_argument('-o', '--output_dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('-ih', '--input_height', type=int, default=128,
                        help='Model input height (must match training)')
    parser.add_argument('-iw', '--input_width', type=int, default=128,
                        help='Model input width (must match training)')

    args = parser.parse_args()

    predict_images(
        args.model, 
        args.input_dir, 
        args.output_dir, 
        args.input_height, 
        args.input_width
    )

if __name__ == "__main__":
    run_main()

