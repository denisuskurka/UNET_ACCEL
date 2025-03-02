#!/usr/bin/env python
"""
A minimal training script for your QKeras-based UNet-light model for segmentation,
without pruning, showing the first image+mask with matplotlib, using a Dice metric.
"""

import os
import time
import tensorflow as tf
import numpy as np
from dataset import get_image_mask_paths, create_dataset
from model import build_model
from loss import bce_dice_loss, dice_coefficient
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
HEIGHT, WIDTH = 128, 128
BATCH_SIZE = 8
N_EPOCHS = 1000
LEARNING_RATE = 1e-3

IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

PRUNING = True

# Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs,
# ending by the 10th epoch
def pruneFunction(layer):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=2000, frequency=50)
        #"pruning_schedule": sparsity.ConstantSparsity(0.5, begin_step=0, frequency=50)
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

split_idx = int(0.9 * n_samples)
NSTEPS = split_idx // BATCH_SIZE
train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

train_ds = create_dataset(train_image_paths, train_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

# Show the first image/mask from the training dataset
train_batch = next(iter(train_ds))
first_img, first_mask = train_batch[0][0], train_batch[1][0]
first_img = first_img.numpy()
first_mask = first_mask.numpy()

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

loss_fn = bce_dice_loss(bce_weight=0.3)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss=loss_fn,
    optimizer=optimizer,
    # 2) Use dice_coefficient metric instead of accuracy
    metrics=[dice_coefficient]
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
    patience=100,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=500,
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
# Save the Final Model
# ----------------------------
model = strip_pruning(model)
model.save('quantized_cnn_model_final.h5')
print("Done.")

# ----------------------------
# Check sparsity
# ----------------------------
def doWeights(model):
    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch") != -1 or len(layer.get_weights()) < 1:
            continue
        weights = layer.weights[0].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        print('Layer {}: % of zeros = {}'.format(layer._name, np.sum(weights == 0) / np.size(weights)))

    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])

    fig = plt.figure(figsize=(10, 10))
    bins = np.linspace(-1.5, 1.5, 50)
    plt.hist(histosW, bins, histtype='stepfilled', stacked=True, label=labelsW, edgecolor='black')
    plt.legend(frameon=False, loc='upper left')
    plt.ylabel('Number of Weights')
    plt.xlabel('Weights')
    plt.figtext(0.2, 0.38, model._name, wrap=True, horizontalalignment='left', verticalalignment='center')

doWeights(model)
