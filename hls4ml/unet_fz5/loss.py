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

# ----------------------------------------------------
# 1) Dice Metric definition
# ----------------------------------------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    A 'soft' Dice metric that compares predictions (logits) vs. ground truth.
    """
    # Convert logits -> probabilities
    y_pred = tf.nn.sigmoid(y_pred)

    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Intersection + union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth

    dice_score = (2.0 * intersection + smooth) / union
    return dice_score
