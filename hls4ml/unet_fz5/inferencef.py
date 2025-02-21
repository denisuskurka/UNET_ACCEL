#!/usr/bin/env python
"""
A simple inference script for the float U-Net model.

Steps:
  • Loads one image from the images folder.
  • Preprocesses it (grayscale, resized to 128×128).
  • Exports the preprocessed image as 'X_test.npy'.
  • Loads the float model (with compile=False to avoid requiring the training loss).
  • Runs inference to get a baseline response.
  • Exports the predicted mask as 'Y_baseline.npy'.
  • Displays the input image and predicted mask using Matplotlib.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Force TensorFlow to use the CPU only (optional).
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

# ----------------------------
# Parameters
# ----------------------------
IMAGE_HEIGHT = 128
IMAGE_WIDTH  = 128
IMAGES_DIR   = "./data/images"            # folder with your images
MODEL_PATH   = "quantized_cnn_model_final.h5"       # path to your saved float model
INPUT_EXPORT = "X_test.npy"               # filename to export input image
OUTPUT_EXPORT = "Y_baseline.npy"          # filename to export model prediction

# ----------------------------
# Utility Functions
# ----------------------------
def load_and_preprocess_image(image_path, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
    """
    Loads an image file, decodes it as a grayscale image,
    converts it to float32 in [0, 1] range, and resizes it.
    """
    image = tf.io.read_file(image_path)
    # Decode as PNG with 1 channel (grayscale)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])
    return image

def get_first_image_path(images_dir, valid_exts=('.png', '.jpg', '.jpeg')):
    """
    Returns the full path of the first image in images_dir that matches a valid extension.
    """
    for fname in sorted(os.listdir(images_dir)):
        if fname.lower().endswith(valid_exts):
            return os.path.join(images_dir, fname)
    return None

def show_result(input_image, pred_mask):
    """
    Displays the input image and predicted mask side-by-side.
    
    Parameters:
      input_image: NumPy array of shape (H, W, 1)
      pred_mask: NumPy array of shape (H, W) or (H, W, 1)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(np.squeeze(input_image), cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(np.squeeze(pred_mask), cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Function
# ----------------------------
def main():
    # Get one image from the images folder.
    image_path = get_first_image_path(IMAGES_DIR)
    if image_path is None:
        print(f"No image files found in {IMAGES_DIR}.")
        return
    print("Using image:", image_path)
    
    # Load and preprocess the image.
    image = load_and_preprocess_image(image_path)
    # Expand dims => shape becomes (1, 128, 128, 1) for batch inference
    image_batch = tf.expand_dims(image, axis=0)
    
    # Export the preprocessed input image as X_test.npy
    np.save(INPUT_EXPORT, image.numpy())
    print(f"Exported preprocessed input image to '{INPUT_EXPORT}' with shape {image.numpy().shape}.")
    
    # Load the float model (no custom objects, no compile).
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded from:", MODEL_PATH)
    
    # Run inference.
    print("Running inference...")
    pred = model.predict(image_batch)
    # Remove batch dimension => shape (128, 128) if (1,128,128,1)
    pred_mask = np.squeeze(pred)
    print("Prediction shape:", pred_mask.shape)
    
    # Export the predicted mask as Y_baseline.npy
    np.save(OUTPUT_EXPORT, pred_mask)
    print(f"Exported model prediction to '{OUTPUT_EXPORT}'.")
    
    # Display the input image and predicted mask
    show_result(image.numpy(), pred_mask)

if __name__ == "__main__":
    main()
