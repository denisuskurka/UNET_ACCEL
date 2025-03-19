import tensorflow as tf

# ----------------------------
# Loss Functions
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

def bce_dice_loss(bce_weight=0.3):
    """
    Returns a combined loss: bce_weight * BCE + (1.0 - bce_weight) * Dice loss.
    
    Assumes y_pred is already in [0,1] => from_logits=False for BCE.
    """
    # Since the model outputs probabilities, do NOT use from_logits=True
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    def loss(y_true, y_pred):
        loss_bce = bce(y_true, y_pred)
        loss_dice = dice_loss(y_true, y_pred)
        return bce_weight * loss_bce + (1.0 - bce_weight) * loss_dice
    return loss

# ----------------------------------------------------
# Dice Metric
# ----------------------------------------------------
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    A 'soft' Dice metric that compares model probabilities vs. ground truth.
    """
    # y_pred is already in [0,1], so we do NOT apply tf.nn.sigmoid here.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    dice_score = (2.0 * intersection + smooth) / union
    return dice_score
