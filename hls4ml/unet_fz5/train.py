#!/usr/bin/env python
"""
A minimal training script for your QKeras-based UNet-light model for segmentation.
This version uses a combined Binary Crossentropy + Dice loss to improve mask generation.
It forces TensorFlow to run on the CPU (to help debug GPU/cuDNN issues).

Requirements:
  • A 'dataset' module that provides get_image_mask_paths() and create_dataset().
  • A 'model' module that provides build_model(HEIGHT, WIDTH), which builds your QKeras model.
"""

import os
import time
import tensorflow as tf
import numpy as np
from dataset import get_image_mask_paths, create_dataset
from model import build_model
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks

# ----------------------------
# Parameters
# ----------------------------
HEIGHT, WIDTH = 128, 128      # image/mask dimensions
BATCH_SIZE = 4               # adjust as needed
n_epochs = 1000
LEARNING_RATE = 0.01          # learning rate

# Directories for your data (make sure these paths are correct)
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

# ----------------------------
# Loss Functions: Dice and Combined BCE + Dice Loss
# ----------------------------
def dice_loss(y_true, y_pred, eps=1e-6):
    """
    Computes the Dice loss. Applies sigmoid to the logits before computing the Dice coefficient.
    
    Parameters:
      y_true: Ground truth mask.
      y_pred: Logits from the model.
      eps: Small epsilon value to avoid division by zero.
    
    Returns:
      Dice loss value.
    """
    y_pred = tf.nn.sigmoid(y_pred)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + eps
    dice = 2.0 * intersection / union
    return 1.0 - dice

def bce_dice_loss(bce_weight=0.3):
    """
    Returns a combined loss function as a weighted sum of Binary Crossentropy (from logits)
    and Dice loss.
    
    Parameters:
      bce_weight: Weight factor for the BCE loss. (Dice weight will be 1.0 - bce_weight)
    
    Returns:
      A loss function that computes: bce_weight * BCE + (1.0 - bce_weight) * Dice loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def loss(y_true, y_pred):
        loss_bce = bce(y_true, y_pred)
        loss_dice = dice_loss(y_true, y_pred)
        return bce_weight * loss_bce + (1.0 - bce_weight) * loss_dice
    return loss

# Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs,
# ending by the 10th epoch
def pruneFunction(layer):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.50, begin_step=NSTEPS * 2, end_step=NSTEPS * 10, frequency=NSTEPS
        )
    }
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer

# ----------------------------
# Prepare Datasets
# ----------------------------
image_paths, mask_paths = get_image_mask_paths(IMAGES_DIR, MASKS_DIR)
n_samples = len(image_paths)
split_idx = int(0.9 * n_samples)  # 70% for training, 30% for validation
NSTEPS = split_idx // BATCH_SIZE

train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

train_ds = create_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

# ----------------------------
# Build the Model
# ----------------------------
model = build_model(HEIGHT, WIDTH)
model = tf.keras.models.clone_model(model, clone_function=pruneFunction)

# ----------------------------
# Compile the Model with Combined BCE + Dice Loss
# ----------------------------
loss_fn = bce_dice_loss(bce_weight=0.3)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

# ----------------------------
# Training
# ----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=200, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    pruning_callbacks.UpdatePruningStep()
]

print("Starting training...")
start = time.time()
history = model.fit(train_ds, epochs=n_epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
end = time.time()
print(f"\nTraining completed in {(end - start) / 60.0:.2f} minutes.")

# Save the trained model
model.save('quantized_cnn_model_cpu.h5')
