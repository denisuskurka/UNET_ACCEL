#!/usr/bin/env python
"""
train.py - Training script for QKeras ellipse regressor with optional pruning.

Features:
  - Predicts 5 ellipse parameters from ultrasound images
  - Uses MSE loss
  - Optionally prunes the model (PRUNING flag)
  - Saves best model + final stripped model
  - Plots loss/MAE curves
  - Prints weight sparsity layer-by-layer
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from dataset import get_image_mask_paths, create_dataset, draw_ellipse_on_blank
from model import build_model  # QKeras model with quantized layers

# ----------------------------
# Parameters
# ----------------------------
HEIGHT, WIDTH = 128, 128
BATCH_SIZE = 4
N_EPOCHS = 1000
LEARNING_RATE = 0.001

IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

PRUNING = True  # Set True to enable model pruning

# Pruning schedule: from 0 to 50% sparsity
def pruneFunction(layer):
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.9,
            begin_step=0,
            end_step=2000,
            frequency=50
        )
    }
    if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer

# ----------------------------
# Data Loading
# ----------------------------
image_paths, mask_paths = get_image_mask_paths(IMAGES_DIR, MASKS_DIR)
n_samples = len(image_paths)
split_idx = int(0.8 * n_samples)

train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

train_ds = create_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

# ----------------------------
# Sanity Check: Visualize Sample
# ----------------------------
image_batch, params_batch = next(iter(train_ds))
image_np = image_batch[0].numpy().squeeze()
params_np = params_batch[0].numpy()
ellipse_img = draw_ellipse_on_blank(HEIGHT, WIDTH, params_np)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_np, cmap='gray')
plt.title("Training Image")

plt.subplot(1, 2, 2)
plt.imshow(ellipse_img, cmap='gray')
plt.title("Reconstructed Ellipse")
plt.tight_layout()
plt.show()

# ----------------------------
# Build & Compile Model
# ----------------------------
model = build_model(HEIGHT, WIDTH)
if PRUNING:
    model = tf.keras.models.clone_model(model, clone_function=pruneFunction)

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mae"]
)

# ----------------------------
# Callbacks
# ----------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=100, restore_best_weights=True, verbose=1
)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=10, verbose=1
)

callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]
if PRUNING:
    callbacks.append(pruning_callbacks.UpdatePruningStep())

# ----------------------------
# Training
# ----------------------------
print("Starting training...")
start = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=N_EPOCHS,
    callbacks=callbacks,
    verbose=1
)
end = time.time()
print(f"\nTraining completed in {(end - start) / 60.0:.2f} minutes.")

# ----------------------------
# Strip pruning and Save
# ----------------------------
model = strip_pruning(model)
model.save('quantized_cnn_model_final.h5')
print("Saved final stripped model -> quantized_cnn_model_final.h5")

# ----------------------------
# Plot training history
# ----------------------------
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_ran = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_ran, train_loss, label="Train Loss")
plt.plot(epochs_ran, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

if 'mae' in history.history:
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, train_mae, label="Train MAE")
    plt.plot(epochs_ran, val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE Over Epochs")
    plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# Weight Sparsity Report
# ----------------------------
def doWeights(a_model):
    allWeightsByLayer = {}
    for layer in a_model.layers:
        if ('batch' in layer.name.lower()) or (len(layer.get_weights()) < 1):
            continue
        w = layer.get_weights()[0]
        weights_flat = w.flatten()
        allWeightsByLayer[layer.name] = weights_flat
        pct_zeros = np.sum(weights_flat == 0) / weights_flat.size
        print(f'Layer {layer.name}: % of zeros = {pct_zeros:.2%}')

    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])

    fig = plt.figure(figsize=(10, 10))
    bins = np.linspace(-1.5, 1.5, 50)
    plt.hist(histosW, bins, histtype='stepfilled', stacked=True,
             label=labelsW, edgecolor='black')
    plt.legend(frameon=False, loc='upper left')
    plt.ylabel('Number of Weights')
    plt.xlabel('Weights')
    plt.figtext(0.2, 0.38, a_model.name, wrap=True,
                horizontalalignment='left', verticalalignment='center')

if PRUNING:
    doWeights(model)
