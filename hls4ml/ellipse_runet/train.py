#!/usr/bin/env python
"""
train.py - Training script for ellipse regressor.

Features:
  - Predicts 5 ellipse parameters from ultrasound images
  - Uses MSE loss
  - Shows first training image with reconstructed ellipse
  - Saves best model and final model
  - Plots training history
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import get_image_mask_paths, create_dataset, draw_ellipse_on_blank
from model import build_model

# ----------------------------
# Parameters
# ----------------------------
HEIGHT, WIDTH = 128, 128
BATCH_SIZE = 4
N_EPOCHS = 1000
LEARNING_RATE = 0.01

IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

# ----------------------------
# Data Loading
# ----------------------------
image_paths, mask_paths = get_image_mask_paths(IMAGES_DIR, MASKS_DIR)
n_samples = len(image_paths)

split_idx = int(0.6 * n_samples)
train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

train_ds = create_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

# ----------------------------
# Check Sample Data
# ----------------------------
image_batch, params_batch = next(iter(train_ds))
image_np = image_batch[0].numpy().squeeze()
params_np = params_batch[0].numpy()

print("Ellipse Params:", params_np)

reconstructed_mask = draw_ellipse_on_blank(HEIGHT, WIDTH, params_np)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_np, cmap='gray')
plt.title("Training Image")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_mask, cmap='gray')
plt.title("Reconstructed Ellipse")

plt.tight_layout()
plt.show()

# ----------------------------
# Build & Compile Model
# ----------------------------
model = build_model(HEIGHT, WIDTH)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mae"]
)

# ----------------------------
# Callbacks
# ----------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    verbose=1
)

callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]

# ----------------------------
# Training
# ----------------------------
print("Starting training...")
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
# Save Final Model
# ----------------------------
model.save('ellipse_regressor_final.h5')
print("Saved final model -> ellipse_regressor_final.h5")

# ----------------------------
# Plot Training History
# ----------------------------
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_ran = range(1, len(train_loss) + 1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_ran, train_loss, label="Train Loss")
plt.plot(epochs_ran, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

if 'mae' in history.history:
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    plt.subplot(1,2,2)
    plt.plot(epochs_ran, train_mae, label="Train MAE")
    plt.plot(epochs_ran, val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE Over Epochs")
    plt.legend()

plt.tight_layout()
plt.show()
