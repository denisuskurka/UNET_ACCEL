#!/usr/bin/env python
"""
A minimal training script for your QKeras model that uses your image/mask data.
This version forces TensorFlow to run on the CPU (to help debug GPU/cuDNN issues).
"""

import os
import time
import tensorflow as tf
import numpy as np
from dataset import get_image_mask_paths, create_dataset
from model import build_model

# Force TensorFlow to use the CPU only.
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

# ----------------------------
# Parameters
# ----------------------------
HEIGHT, WIDTH = 128, 128     # image/mask dimensions
BATCH_SIZE = 16               # adjust as needed
n_epochs = 100
LEARNING_RATE = 3e-3         # learning rate

# Directories for your data (make sure these paths are correct)
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

# ----------------------------
# Prepare Datasets
# ----------------------------
image_paths, mask_paths = get_image_mask_paths(IMAGES_DIR, MASKS_DIR)
n_samples = len(image_paths)
split_idx = int(0.7 * n_samples)  # 70% for training, 30% for validation

train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

train_ds = create_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

model = build_model(HEIGHT, WIDTH)

# ----------------------------
# Compile the Model
# ----------------------------
# Using BinaryCrossentropy (from_logits=True) for this segmentation-style task.
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

# ----------------------------
# Training
# ----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("Starting training on CPU...")
start = time.time()
history = model.fit(train_ds, epochs=n_epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
end = time.time()
print(f"\nTraining completed in {(end - start) / 60.0:.2f} minutes.")

# Save the trained model
model.save('quantized_cnn_model_cpu.h5')
