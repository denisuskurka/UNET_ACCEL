#!/usr/bin/env python
"""
A simple inference script for a QKeras model, enhanced to display negative vs. positive
logits in a diverging colormap with a colorbar.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from loss import bce_dice_loss, dice_coefficient

# Import the needed QKeras layers for custom objects.
from qkeras import QConv2DBatchnorm, QActivation

# Force TensorFlow to use the CPU only (optional).
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

# ----------------------------
# Parameters
# ----------------------------
IMAGE_HEIGHT = 64
IMAGE_WIDTH  = 64
IMAGES_DIR   = "./data/images"            
MODEL_PATH   = "quantized_cnn_model_final.h5"    
INPUT_EXPORT = "X_test.npy"                
OUTPUT_EXPORT = "Y_baseline.npy"           

# ----------------------------
# Utility Functions
# ----------------------------
def load_and_preprocess_image(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    Loads an image file, decodes it as a grayscale image,
    converts it to float32 in [0, 1] range, and resizes it.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)       # grayscale
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])
    return image

def get_first_image_path(images_dir, valid_exts=('.png', '.jpg', '.jpeg')):
    """Returns the full path of the first valid image file in images_dir."""
    for fname in sorted(os.listdir(images_dir)):
        if fname.lower().endswith(valid_exts):
            return os.path.join(images_dir, fname)
    return None

def show_result(input_image, pred_mask):
    """
    Displays the input image in grayscale and the predicted mask with a diverging colormap,
    so negative values appear different from positive values. A colorbar is included to show scale.
    
    Parameters:
      input_image: NumPy array of shape (H, W, 1)
      pred_mask: NumPy array of shape (H, W) or (H, W, 1)
    """
    # Flatten the extra dimension if present
    mask_2d = np.squeeze(pred_mask)
    
    # Determine the range so that negative is distinctly visible from positive
    abs_max = max(abs(mask_2d.min()), abs(mask_2d.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: input image in grayscale
    axes[0].imshow(np.squeeze(input_image), cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Right: predicted mask with a diverging colormap (e.g. 'seismic')
    im = axes[1].imshow(
        mask_2d,
        cmap='seismic',           # or 'bwr', 'RdBu', etc.
        vmin=-abs_max,
        vmax=abs_max
    )
    axes[1].set_title("Predicted Mask (Logits)")
    axes[1].axis("off")

    # Add a colorbar for the mask
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Function
# ----------------------------
def main():
    image_path = get_first_image_path(IMAGES_DIR)
    if image_path is None:
        print(f"No image files found in {IMAGES_DIR}.")
        return
    print("Using image:", image_path)
    
    # Load and preprocess the image (H, W, 1)
    image = load_and_preprocess_image(image_path)
    # Shape => (1, H, W, 1) for batch inference
    image_batch = tf.expand_dims(image, axis=0)
    
    # Export the preprocessed input image as X_test.npy.
    np.save(INPUT_EXPORT, image.numpy())
    print(f"Exported preprocessed input image to '{INPUT_EXPORT}' with shape {image.numpy().shape}.")
    
    # Load the QKeras model with a custom object scope so QKeras layers are recognized.
    # compile=False to avoid reloading the custom loss.
    co = {
        "loss": bce_dice_loss(bce_weight=0.3),
        "dice_coefficient": dice_coefficient
    }
    _add_supported_quantized_objects(co)
    co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=co, compile=False)
    print("Model loaded from:", MODEL_PATH)
    
    # Run inference
    print("Running inference...")
    pred = model.predict(image_batch)
    pred_mask = np.squeeze(pred)  # (H, W) if batch=1
    print("Prediction shape:", pred_mask.shape)
    
    # Export the predicted mask as Y_baseline.npy.
    np.save(OUTPUT_EXPORT, pred_mask)
    print(f"Exported model prediction to '{OUTPUT_EXPORT}'.")
    
    # Show the input image & predicted mask
    show_result(image.numpy(), pred_mask)

if __name__ == "__main__":
    main()
