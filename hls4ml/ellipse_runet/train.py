#!/usr/bin/env python
"""
train.py - A minimal training script for your reduced QKeras-based U-Net model for segmentation.

Features:
  - Uses either BCE+Dice or Focal Tversky loss from 'loss.py'
  - Displays the first training image & mask
  - Optionally prunes the model (PRUNING flag)
  - Plots training/validation loss over epochs
  - Saves the best model to 'best_model.h5'
  - Saves a final stripped/pruned model to 'quantized_cnn_model_final.h5'

Requirements:
  • 'dataset' module providing get_image_mask_paths(...) and create_dataset(...)
  • 'model' module providing build_model(...)
  • 'loss.py' module containing bce_dice_loss, dice_loss, focal_tversky_loss
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning

# Our local modules
from dataset import get_image_mask_paths, create_dataset
from model import build_model
from loss import bce_dice_loss, focal_tversky_loss

# ----------------------------
# Parameters
# ----------------------------
HEIGHT, WIDTH = 128, 128
BATCH_SIZE = 8
N_EPOCHS = 1000
LEARNING_RATE = 0.01

IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

PRUNING = True  # Set True if you want to prune
# Choose which loss function to use:
USE_FOCAL_TVERSKY = True  # True => use focal_tversky_loss; False => bce_dice_loss

# Pruning schedule: from 0 to 50% sparsity, etc.
def pruneFunction(layer):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=20000,
            frequency=200
        )
    }
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer

# ----------------------------
# Data Loading
# ----------------------------
image_paths, mask_paths = get_image_mask_paths(IMAGES_DIR, MASKS_DIR)
n_samples = len(image_paths)

# 80% train, 20% validation
split_idx = int(0.8 * n_samples)
train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

train_ds = create_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

# Show the first image/mask from the training dataset
train_batch = next(iter(train_ds))
first_img, first_mask = train_batch[0][1], train_batch[1][1]
first_img = first_img.numpy()
first_mask = first_mask.numpy()

print("Mask min/max:", first_mask.min(), first_mask.max())
print("Image min/max:", first_img.min(), first_img.max())
print("first_img shape:", first_img.shape)
print("first_mask shape:", first_mask.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.imshow(first_img[..., 0], cmap='gray')
ax1.set_title("First Training Image")
ax1.axis('off')

ax2.imshow(first_mask[..., 0], cmap='gray')
ax2.set_title("First Training Mask")
ax2.axis('off')
plt.tight_layout()
plt.show()

# ----------------------------
# Build & Compile Model
# ----------------------------
model = build_model(HEIGHT, WIDTH)
if PRUNING:
    model = tf.keras.models.clone_model(model, clone_function=pruneFunction)

# Choose your loss function:
if USE_FOCAL_TVERSKY:
    print("Using Focal Tversky Loss...")
    loss_fn = focal_tversky_loss(alpha=0.3, beta=0.7, gamma=3.0)
else:
    print("Using BCE+Dice Loss...")
    loss_fn = bce_dice_loss(bce_weight=0.3)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# We'll just track "accuracy" or you can leave it empty if you prefer
model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

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
    patience=200,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=20,
    verbose=1
)

callbacks = []
if PRUNING:
    callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb, pruning_callbacks.UpdatePruningStep()]
else:
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
# Strip pruning (if used), then Save the Final Model
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

plt.figure(figsize=(12,5))

# Left: Loss
plt.subplot(1,2,1)
plt.plot(epochs_ran, train_loss, label="Train Loss")
plt.plot(epochs_ran, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

# Right: Accuracy (optional)
if 'accuracy' in history.history:
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.subplot(1,2,2)
    plt.plot(epochs_ran, train_acc, label="Train Acc")
    plt.plot(epochs_ran, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# Optional: Check weight sparsity if pruning was used
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
