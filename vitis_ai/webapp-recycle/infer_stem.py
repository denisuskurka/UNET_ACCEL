#!/usr/bin/env python
import time
import os
import tensorflow as tf
import numpy as np
from qkeras.utils import _add_supported_quantized_objects
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

# Force TensorFlow to use the CPU only (optional).
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

stem_model = None

IMAGE_HEIGHT = 128
IMAGE_WIDTH  = 128

MODEL_PATH    = "runet_stem_model.h5"
RAW_EXPORT    = "./data/data_stem_input.bin"
OUTPUT_EXPORT = "Y_baseline.npy"

def encode(xi):
    return np.int32(round(xi * 2**24)) # note 2**10 = 2**(A-B)
def decode(yi):
    return yi * 2**-8
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

def load_stem_model():
    global stem_model
    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    custom_objects['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=custom_objects, compile=False
    )
    print("Model loaded from:", MODEL_PATH)
    stem_model = model

def infer_stem(cropped_filepath, hw=False):
    global stem_model
    # Load cropped file
    image_path = cropped_filepath
    if image_path is None:
        print(f"No image files found in {cropped_filepath}.")
        return
    print("Using image:", image_path)

    # Load and preprocess the image => shape (H, W, 1), float32
    image = load_and_preprocess_image(image_path)
    # Expand batch dimension => (1, H, W, 1)
    image_batch = tf.expand_dims(image, axis=0)

    image_raw = image_batch.numpy().astype(np.float32)  # shape (1, H, W, 1)
    image_raw.ravel().astype(np.float32).tofile(RAW_EXPORT)

    # Load model into global var stem_model
    if(stem_model is None):
        load_stem_model()

    print(stem_model.input.dtype)
    print(stem_model.output.dtype)
    print("image_raw min/max:", image_raw.min(), image_raw.max())
    print("encoded min/max:", encode_v(image_raw).min(), encode_v(image_raw).max())

    if(hw):
        os.system('./run_dma.sh')
        time.sleep(1)
        count = 0
        while not os.path.exists("./data/result.bin"):
            print("Waiting for DMA...")
            time.sleep(1)
            count = count + 1
            if count > 10:
                print("DMA FAILED!")
                break
        data_float = np.fromfile("./data/result.bin", dtype=np.float32)
        return data_float
    else:
        # Run inference
        print("Running inference...")
        pred = stem_model.predict(image_batch)
        pred_mask = np.squeeze(pred)

        return pred_mask
