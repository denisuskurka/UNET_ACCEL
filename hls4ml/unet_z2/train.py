#!/usr/bin/env python
# File: train.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
A minimal training script for your QKeras-based UNet-light model for segmentation.
This version uses Focal Tversky loss to better handle small masks (~1% of the image).

Changes:
  - Only saves the best model (by val_loss) as "best_model.h5"
  - At the end, plots training history (loss & accuracy) using matplotlib

Requirements:
  窶｢ A 'dataset' module that provides get_image_mask_paths() and create_dataset().
  窶｢ A 'model' module that provides build_model(HEIGHT, WIDTH), which builds your QKeras model.
"""

import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_image_mask_paths, create_dataset
from model import build_model

# ----------------------------
# Parameters
# ----------------------------
HEIGHT, WIDTH = 128, 128       # image/mask dimensions
BATCH_SIZE = 4                 # adjust as needed
N_EPOCHS = 1000
LEARNING_RATE = 0.01           # learning rate

# Directories for your data
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks_outline"

# ----------------------------
# Focal Tversky Loss
# ----------------------------
def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=2.0, eps=1e-6):
    """
    Focal Tversky loss for imbalanced segmentation (esp. small masks).
    
    alpha > 0.5 => weigh FN more
    beta  > 0.5 => weigh FP more (less common for small masks)
    gamma > 1   => focal effect focusing on hard examples

    y_true, y_pred shapes: [batch, height, width, 1]
    y_pred is expected to be probabilities in [0,1].
    """
    def loss(y_true, y_pred):
        # clip to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)

        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # TPs, FPs, FNs
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)

        # Tversky index
        tversky_index = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
        # Focal Tversky
        focal_tversky = tf.pow((1.0 - tversky_index), gamma)

        return focal_tversky
    return loss

# ----------------------------
# Prepare Datasets
# ----------------------------
image_paths, mask_paths = get_image_mask_paths(IMAGES_DIR, MASKS_DIR)
n_samples = len(image_paths)
split_idx = int(0.8 * n_samples)  # 80% for training, 20% for validation

train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

train_ds = create_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

# ----------------------------
# Build the Model
# ----------------------------
model = build_model(HEIGHT, WIDTH)

# ----------------------------
# Compile the Model (Focal Tversky)
# ----------------------------
loss_fn = focal_tversky_loss(alpha=0.3, beta=0.7, gamma=3.0)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

# ----------------------------
# Training Callbacks
# ----------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",       # save as "best_model.h5"
    monitor="val_loss",    # track validation loss
    save_best_only=True,   # only save the best model
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    patience=10,
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]

# ----------------------------
# Train the Model
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
# We do NOT save the final model. We rely on best_model.h5
# ----------------------------

# ----------------------------
# Summarize and Plot History
# ----------------------------
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs_ran = range(1, len(train_loss) + 1)

print("\nFinal Epoch Stats:")
print(f"  Training Loss:      {train_loss[-1]:.4f}")
print(f"  Validation Loss:    {val_loss[-1]:.4f}")
print(f"  Training Accuracy:  {train_acc[-1]:.4f}")
print(f"  Validation Accuracy:{val_acc[-1]:.4f}")

# Plot the loss curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_ran, train_loss, label='Train Loss')
plt.plot(epochs_ran, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot the accuracy curves
plt.subplot(1,2,2)
plt.plot(epochs_ran, train_acc, label='Train Acc')
plt.plot(epochs_ran, val_acc, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

print("\nDone. The best model is saved as 'best_model.h5'.")

