# File: hls4ml/unet_z2/dataset.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import time
import tensorflow as tf
import numpy as np

# ----------------------------
# Data Loading Functions
# ----------------------------
def get_image_mask_paths(images_dir, masks_dir):
    """List and sort image and mask file paths. If counts differ, use the first min(count) pairs."""
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_paths = sorted([
        os.path.join(images_dir, fname)
        for fname in os.listdir(images_dir)
        if fname.lower().endswith(valid_exts)
    ])
    mask_paths = sorted([
        os.path.join(masks_dir, fname)
        for fname in os.listdir(masks_dir)
        if fname.lower().endswith(valid_exts)
    ])
    min_count = min(len(image_paths), len(mask_paths))
    print(f"Found {len(image_paths)} images and {len(mask_paths)} masks. Using {min_count} pairs.")
    return image_paths[:min_count], mask_paths[:min_count]

def parse_image_mask(image_path, mask_path, HEIGHT, WIDTH):
    """Reads an image and mask file, decodes them, and resizes to the target dimensions."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, [HEIGHT, WIDTH])
    
    return image, mask

def create_dataset(image_paths, mask_paths, batch_size, HEIGHT, WIDTH):
    """Creates a tf.data.Dataset from image and mask paths."""
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(lambda ip, mp: parse_image_mask(ip, mp, HEIGHT, WIDTH),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
