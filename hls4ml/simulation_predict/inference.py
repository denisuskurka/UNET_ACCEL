#!/usr/bin/env python
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2DBatchnorm, QActivation

# Force TensorFlow to use the CPU only (optional).
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

IMAGE_HEIGHT = 128
IMAGE_WIDTH  = 128
IMAGES_DIR   = "./data/images"

#MODEL_PATH    = "quantized_cnn_model_final.h5"
MODEL_PATH    = "best_model.h5"
INPUT_EXPORT  = "X_test.npy"     # still save as .npy
RAW_EXPORT    = "X_test.bin"     # new: raw binary file
OUTPUT_EXPORT = "Y_baseline.npy"

def encode(xi):
    return np.int32(round(xi * 2**24)) # note 2**10 = 2**(A-B)
def decode(yi):
    return yi * 2**-24
encode_v = np.vectorize(encode) # to apply them element-wise
decode_v = np.vectorize(decode)

def load_and_preprocess_image(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    Loads an image file, decodes it as a grayscale image,
    converts it to float32 in [0, 1] range, and resizes it.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # grayscale
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])
    return image

def get_image_x_path(images_dir, x=1, valid_exts=('.png', '.jpg', '.jpeg')):
    """Returns the full path of the x-th valid image file in images_dir.
    
    Args:
        images_dir (str): The directory to search for images.
        x (int): The index (1-based) of the image to return.
        valid_exts (tuple): Tuple of valid file extensions.
    
    Returns:
        str or None: The full path of the x-th valid image, or None if not found.
    """
    images = sorted(
        [fname for fname in os.listdir(images_dir) if fname.lower().endswith(valid_exts)]
    )
    
    if 1 <= x <= len(images):
        return os.path.join(images_dir, images[x - 1])
    return None

def show_result(input_image, pred_mask):
    """
    Displays the input image in grayscale and the predicted mask with a diverging colormap,
    so negative values appear different from positive values. A colorbar is included to show scale.
    """
    mask_2d = np.squeeze(pred_mask)
    abs_max = max(abs(mask_2d.min()), abs(mask_2d.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(np.squeeze(input_image), cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(
        mask_2d,
        cmap='seismic',
        vmin=-abs_max,
        vmax=abs_max
    )
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def main():
    image_path = get_image_x_path(IMAGES_DIR, 2)
    if image_path is None:
        print(f"No image files found in {IMAGES_DIR}.")
        return
    print("Using image:", image_path)

    # Load and preprocess the image => shape (H, W, 1), float32
    image = load_and_preprocess_image(image_path)
    image1 = np.ascontiguousarray(image)
    image1.tofile("X_test1.bin")
    # Expand batch dimension => (1, H, W, 1)
    image_batch = tf.expand_dims(image, axis=0)

    # ----------------------------------------------------------------------
    # 1. SAVE AS .NPY
    # ----------------------------------------------------------------------
    np.save(INPUT_EXPORT, image.numpy())
    print(f"Exported preprocessed input image to '{INPUT_EXPORT}' with shape {image.shape}.")

    # ----------------------------------------------------------------------
    # 2. SAVE AS RAW BINARY (no header), FIXED32,8
    #    This is the file you can dd into memory as raw bytes.
    # ----------------------------------------------------------------------
    # Convert to float32 array explicitly (should already be float32,
    # but let's be sure).
    image_raw = image.numpy().astype(np.float32)  # shape (H, W, 1)
    # apply your vectorized encode
    image_fixed = encode_v(image_raw)  # shape (H, W, 1), int32
    # write out the integer data as raw bytes
    image_fixed.tofile(RAW_EXPORT)
    #image_raw.tofile(RAW_EXPORT)

    print(f"Also exported raw bytes to '{RAW_EXPORT}' "
          f"(size: {image_raw.size} floats => {image_raw.size * 4} bytes).")

    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    custom_objects['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=custom_objects, compile=False
    )
    print("Model loaded from:", MODEL_PATH)

    # Run inference
    print("Running inference...")
    pred = model.predict(image_batch)
    pred_mask = np.squeeze(pred)
    print("Prediction shape:", pred_mask.shape)

    # Save predicted mask as .npy
    np.save(OUTPUT_EXPORT, pred_mask)
    print(f"Exported model prediction to '{OUTPUT_EXPORT}'.")

    # Display input vs. prediction
    show_result(image.numpy(), pred_mask)

if __name__ == "__main__":
    main()
