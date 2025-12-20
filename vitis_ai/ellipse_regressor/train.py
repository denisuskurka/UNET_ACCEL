#!/usr/bin/env python
# File: vitis_ai/ellipse_regressor/train.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
train.py - Training script for Ellipse Regression.

Features:
  - Generates labels on-the-fly by fitting ellipses to mask images (OpenCV).
  - Uses the 'ellipse_regressor' model (Regression CNN).
  - Uses Mean Squared Error (MSE) Loss.
  - Robustly matches images and masks by filename to avoid dimension errors.
"""

import os
import time
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Force CPU usage (often cleaner for Vitis AI unless GPU env is perfect)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. Force eager execution (Essential for Vitis AI TF 1.x environments)
if tf.__version__.startswith('1.'):
    tf.compat.v1.enable_eager_execution()
    print("[INFO] TensorFlow 1.x detected. Eager execution enabled.")

# Import the regression model
from model import build_ellipse_regressor

DIVIDER = '-----------------------------------------'

def get_ellipse_labels(mask_path_tensor):
    """
    Python function to be wrapped in tf.py_function.
    Reads a mask image path, calculates ellipse params, and returns them.
    """
    mask_path = mask_path_tensor.numpy().decode('utf-8')
    
    # Read mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)

    # Threshold to binary
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    # Handle OpenCV version differences for findContours return values
    cnts_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    if len(contours) > 0:
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            # Fit ellipse
            ((cx, cy), (w, h), angle) = cv2.fitEllipse(c)
            # Return params: cx, cy, semi-axis1, semi-axis2, angle
            # We divide w/h by 2 because fitEllipse returns full diameters
            return np.array([cx, cy, w/2.0, h/2.0, angle], dtype=np.float32)
            
    # Default if no ellipse found
    return np.array([0, 0, 0, 0, 0], dtype=np.float32)


def process_path(image_path, mask_path, img_height, img_width):
    """
    TF Data Pipeline function.
    Loads Image (Tensor) and generates Label (Vector) from Mask Path.
    """
    # 1. Load Image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32) # [0, 1]
    img = tf.image.resize(img, [img_height, img_width])
    
    # 2. Generate Label (Ellipse Params) from Mask Path
    # We must use py_function because cv2.fitEllipse is not a TF op
    label = tf.py_function(
        func=get_ellipse_labels,
        inp=[mask_path],
        Tout=tf.float32
    )
    
    # Explicitly set shape because py_function loses shape info
    label.set_shape([5]) 
    
    return img, label


def create_regression_dataset(image_paths, mask_paths, batch_size, height, width):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Map the loading function
    ds = ds.map(
        lambda x, y: process_path(x, y, height, width),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def train(input_height, input_width, batchsize, learnrate, epochs, keras_hdf5, tboard):
    # ----------------------------
    # Parameters
    # ----------------------------
    HEIGHT, WIDTH = input_height, input_width
    BATCH_SIZE = batchsize
    N_EPOCHS = epochs
    LEARNING_RATE = learnrate

    IMAGES_DIR = "./data/images"
    MASKS_DIR = "./data/masks"

    # ----------------------------
    # Data Loading (Robust Matching)
    # ----------------------------
    # Get set of filenames to find the intersection
    img_filenames = set(os.listdir(IMAGES_DIR))
    msk_filenames = set(os.listdir(MASKS_DIR))

    # Find files that exist in BOTH folders
    valid_filenames = sorted(list(img_filenames.intersection(msk_filenames)))
    
    # Filter only for .png
    valid_filenames = [f for f in valid_filenames if f.endswith('.png')]

    if len(valid_filenames) == 0:
        print(f"ERROR: No matching files found!")
        print(f"  Images in {IMAGES_DIR}: {len(img_filenames)}")
        print(f"  Masks in {MASKS_DIR}: {len(msk_filenames)}")
        return

    print(f"Found {len(img_filenames)} images and {len(msk_filenames)} masks.")
    print(f"Using {len(valid_filenames)} matched pairs.")

    # Create the full paths lists based on the matched filenames
    all_image_paths = [os.path.join(IMAGES_DIR, f) for f in valid_filenames]
    all_mask_paths  = [os.path.join(MASKS_DIR, f)  for f in valid_filenames]
    
    # Split 80/20
    split_idx = int(0.8 * len(valid_filenames))
    
    train_img_paths = all_image_paths[:split_idx]
    train_msk_paths = all_mask_paths[:split_idx]
    val_img_paths = all_image_paths[split_idx:]
    val_msk_paths = all_mask_paths[split_idx:]

    train_ds = create_regression_dataset(train_img_paths, train_msk_paths, BATCH_SIZE, HEIGHT, WIDTH)
    val_ds = create_regression_dataset(val_img_paths, val_msk_paths, BATCH_SIZE, HEIGHT, WIDTH)

    # ----------------------------
    # Sanity Check: Visualize Labels
    # ----------------------------
    print("Visualizing first batch ground truth...")
    for imgs, labels in train_ds.take(1):
        # Grab first item
        img_np = imgs[0].numpy()
        label_np = labels[0].numpy()
        
        print(f"Image Shape: {img_np.shape}")
        print(f"Label Vector: {label_np} (cx, cy, axis1, axis2, angle)")

        # Draw ellipse on image for verification
        vis_img = (img_np * 255).astype(np.uint8)
        # Handle grayscale to BGR conversion manually for matplotlib/opencv
        if vis_img.shape[-1] == 1:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        cx, cy, ax1, ax2, angle = label_np
        center = (int(cx), int(cy))
        axes = (int(ax1), int(ax2))
        
        cv2.ellipse(vis_img, center, axes, float(angle), 0, 360, (0, 255, 0), 2)
        
        plt.imshow(vis_img)
        plt.title(f"Ground Truth Check\n{label_np}")
        plt.axis('off')
        plt.show()
        break


    # ----------------------------
    # Build & Compile Model
    # ----------------------------
    model = build_ellipse_regressor(HEIGHT, WIDTH)

    # Loss: MSE is standard for coordinate regression
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Metric: MAE (Mean Absolute Error) is easier to interpret (pixels off)
    metrics = [tf.keras.metrics.MeanAbsoluteError()]

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

    # ----------------------------
    # Callbacks
    # ----------------------------
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        keras_hdf5,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        verbose=1,
        min_lr=1e-6
    )
    
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )

    callbacks = [checkpoint_cb, reduce_lr_cb, early_stop_cb]

    # ----------------------------
    # Training
    # ----------------------------
    print("Starting regression training...")
    start = time.time()
    history = model.fit(
        train_ds,
        epochs=N_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    end = time.time()

    print(f"\nTraining completed in {(end - start) / 60.0:.2f} minutes.")

    # ----------------------------
    # Plot History
    # ----------------------------
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']

    epochs_range = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Train MSE')
    plt.plot(epochs_range, val_loss, label='Val MSE')
    plt.title('Loss (MSE)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, mae, label='Train MAE')
    plt.plot(epochs_range, val_mae, label='Val MAE')
    plt.title('Mean Absolute Error (Pixels)')
    plt.legend()
    
    plt.show()

def run_main():
    print('\n'+DIVIDER)
    print('TensorFlow version : ',tf.__version__)
    print(DIVIDER)

    ap = argparse.ArgumentParser()
    ap.add_argument('-ih', '--input_height', type=int, default=256)
    ap.add_argument('-iw', '--input_width', type=int, default=256)
    ap.add_argument('-b', '--batchsize', type=int, default=32)
    ap.add_argument('-e', '--epochs', type=int, default=100)
    ap.add_argument('-lr', '--learnrate', type=float, default=0.001)
    ap.add_argument('-kh', '--keras_hdf5', type=str, default='ellipse_regressor.h5')
    ap.add_argument('-tb', '--tboard', type=str, default='./tb_logs')    
    args = ap.parse_args()

    train(args.input_height, args.input_width, args.batchsize, 
          args.learnrate, args.epochs, args.keras_hdf5, args.tboard)


if __name__ == '__main__':
    run_main()