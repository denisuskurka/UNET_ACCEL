#!/usr/bin/env python
# File: train.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
train.py - A minimal training script for your reduced QKeras-based U-Net model for segmentation.

Features:
  - Uses either BCE+Dice or Focal Tversky loss from 'loss.py'
  - Displays the first training image & mask
  - Plots training/validation loss over epochs
  - Saves the best model to '<FINAL_NAME>.h5'

Requirements:
  窶｢ 'dataset' module providing get_image_mask_paths(...) and create_dataset(...)
  窶｢ 'model' module providing build_model(...)
  窶｢ 'loss.py' module containing bce_dice_loss, dice_loss, focal_tversky_loss
"""

import os
import time
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# Our local modules
from dataset import get_image_mask_paths, create_dataset
from model import build_model
from loss import bce_dice_loss, focal_tversky_loss

DIVIDER = '-----------------------------------------'

def train(input_height,input_width,batchsize,learnrate,epochs,keras_hdf5,tboard):
    # ----------------------------
    # Parameters
    # ----------------------------
    HEIGHT, WIDTH = input_height, input_width
    BATCH_SIZE = batchsize
    N_EPOCHS = epochs
    LEARNING_RATE = learnrate

    FINAL_NAME = keras_hdf5
    IMAGES_DIR = "./data/images"
    MASKS_DIR = "./data/masks"

    # Choose which loss function to use:
    USE_FOCAL_TVERSKY = True  # True => use focal_tversky_loss; False => bce_dice_loss

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

    # ----------------------------
    # Show the first image/mask from the training dataset
    # ----------------------------
    for train_batch in train_ds:
        first_img, first_mask = train_batch[0][1], train_batch[1][1]
        first_img = first_img.numpy()
        first_mask = first_mask.numpy()
        break

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
        FINAL_NAME,
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

def run_main():
    
    print('\n'+DIVIDER)
    print('Keras version      : ',tf.keras.__version__)
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print(DIVIDER)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-ih', '--input_height',
                    type=int,
                    default='32',
    	            help='Input image height in pixels.')
    ap.add_argument('-iw', '--input_width',
                    type=int,
                    default='32',
    	            help='Input image width in pixels.')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=100,
    	            help='Training batchsize. Must be an integer. Default is 100.')
    ap.add_argument('-e', '--epochs',
                    type=int,
                    default=300,
    	            help='number of training epochs. Must be an integer. Default is 300.')
    ap.add_argument('-lr', '--learnrate',
                    type=float,
                    default=0.001,
    	            help='optimizer initial learning rate. Must be floating-point value. Default is 0.001')
    ap.add_argument('-kh', '--keras_hdf5',
                    type=str,
                    default='./model.hdf5',
    	            help='path of Keras HDF5 file - must include file name. Default is ./model.hdf5')
    ap.add_argument('-tb', '--tboard',
                    type=str,
                    default='./tb_logs',
    	            help='path to folder for saving TensorBoard data. Default is ./tb_logs.')    
    args = ap.parse_args()


    print(' Command line options:')
    print ('--input_height : ',args.input_height)
    print ('--input_width  : ',args.input_width)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print ('--keras_hdf5   : ',args.keras_hdf5)
    print ('--tboard       : ',args.tboard)
    print(DIVIDER)

    train(args.input_height,args.input_width,args.batchsize,args.learnrate,args.epochs,args.keras_hdf5,args.tboard)


if __name__ == '__main__':
    run_main()


