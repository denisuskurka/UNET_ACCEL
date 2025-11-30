#!/usr/bin/env python
"""
Training script for Standard Float U-Net (Vitis AI Ready).

Features:
  - Uses Focal Tversky loss for class imbalance.
  - Saves "best_model.h5" during training (includes optimizer/loss for resuming).
  - Saves "float_model_for_vitis.h5" at the end (weights + arch ONLY, clean for quantization).
  - Plots training history.

Requirements:
  • 'dataset.py': provides get_image_mask_paths() and create_dataset().
  • 'model.py': provides build_model(HEIGHT, WIDTH) -> Standard Keras Model.
"""

import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_image_mask_paths, create_dataset
from model import build_model
from loss import focal_tversky_loss

# ----------------------------
# Parameters
# ----------------------------
HEIGHT, WIDTH = 256, 256       # image/mask dimensions
BATCH_SIZE = 16                # Increased batch size (standard float models usually handle >4 fine)
N_EPOCHS = 50
LEARNING_RATE = 0.001          # Reduced slightly for stability with Adam

# Directories for your data
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

# ----------------------------
# Prepare Datasets
# ----------------------------
# Check if directories exist
if not os.path.exists(IMAGES_DIR) or not os.path.exists(MASKS_DIR):
    print(f"Error: Data directories not found at {IMAGES_DIR} or {MASKS_DIR}")
    exit(1)

image_paths, mask_paths = get_image_mask_paths(IMAGES_DIR, MASKS_DIR)
n_samples = len(image_paths)

if n_samples == 0:
    print("Error: No images found.")
    exit(1)

split_idx = int(0.8 * n_samples)  # 80% training, 20% validation

train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

print(f"Training samples: {len(train_image_paths)}")
print(f"Validation samples: {len(val_image_paths)}")

train_ds = create_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

# ----------------------------
# Build the Model
# ----------------------------
# Ensure this imports the NEW standard float model, not the QKeras one
model = build_model(HEIGHT, WIDTH) 

# ----------------------------
# Compile the Model
# ----------------------------
# alpha=0.3, beta=0.7 penalizes False Positives more (good for precision)
# If you miss too many objects, swap to alpha=0.7, beta=0.3
loss_fn = focal_tversky_loss(alpha=0.3, beta=0.7, gamma=3.0)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

# ----------------------------
# Training Callbacks
# ----------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model_checkpoint.h5",  # Temporary checkpoint with optimizer state
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,    # Increased patience slightly
    restore_best_weights=True, # Crucial: ensures 'model' has best weights at the end
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]

# ----------------------------
# Train the Model
# ----------------------------
print("\nStarting training (Standard Float Model)...")
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
# Save for Vitis AI
# ----------------------------
# Vitis AI prefers a clean model without custom loss functions attached
# This file is the one you will pass to the Quantizer
vitis_model_name = "float_model_for_vitis.h5"
model.save(vitis_model_name, include_optimizer=False)
print(f"\n[IMPORTANT] Clean model for Vitis AI saved as: {vitis_model_name}")

# ----------------------------
# Summarize and Plot History
# ----------------------------
if history.history:
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_ran = range(1, len(train_loss) + 1)

    # Plot
    plt.figure(figsize=(12,5))
    
    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs_ran, train_loss, label='Train Loss')
    plt.plot(epochs_ran, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Focal Tversky Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs_ran, train_acc, label='Train Acc')
    plt.plot(epochs_ran, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Training didn't produce history (maybe stopped immediately).")

print("Done.")
