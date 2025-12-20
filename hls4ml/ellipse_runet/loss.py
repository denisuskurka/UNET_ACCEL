# File: hls4ml/ellipse_runet/loss.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import tensorflow as tf

# ----------------------------
# 1) Dice Loss
# ----------------------------
def dice_loss(y_true, y_pred, eps=1e-6):
    """
    Computes the Dice loss assuming y_pred is already in [0,1].
    (No tf.nn.sigmoid here, since the model output already has a final sigmoid.)
    
    Parameters:
      y_true: Ground truth mask.
      y_pred: Probability map from the model in [0,1].
      eps: Small epsilon value to avoid division by zero.
    """
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + eps
    dice = 2.0 * intersection / union
    return 1.0 - dice

# ----------------------------
# 2) BCE + Dice Loss
# ----------------------------
def bce_dice_loss(bce_weight=0.3):
    """
    Returns a combined loss: bce_weight * BCE + (1.0 - bce_weight) * Dice loss.
    
    Assumes y_pred is already in [0,1] => from_logits=False for BCE.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    def loss(y_true, y_pred):
        loss_bce = bce(y_true, y_pred)
        loss_dice = dice_loss(y_true, y_pred)
        return bce_weight * loss_bce + (1.0 - bce_weight) * loss_dice
    return loss

# ----------------------------
# 3) Focal Tversky Loss
# ----------------------------
def focal_tversky_loss(alpha=0.7, beta=0.3, gamma=2.0, eps=1e-6):
    """
    Focal Tversky loss for imbalanced segmentation (esp. small masks).

    alpha > 0.5 => weigh FN more
    beta  > 0.5 => weigh FP more (less common for small masks)
    gamma > 1   => focal effect, focusing on hard examples

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

# ----------------------------------------------------
# 4) Dice Metric
# ----------------------------------------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    A 'soft' Dice metric that compares model probabilities vs. ground truth.
    """
    # y_pred is already in [0,1], so do NOT apply tf.nn.sigmoid here.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    dice_score = (2.0 * intersection + smooth) / union
    return dice_score
