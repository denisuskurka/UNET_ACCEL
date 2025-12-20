# File: vitis_ai/ellipse_regressor/dataset.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
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

def parse_image_mask(image_path, mask_path, HEIGHT, WIDTH, augment=False):
    """Reads an image and mask file, decodes them, and resizes to the target dimensions."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, [HEIGHT, WIDTH])
    
    if augment:
        image, mask = augment_image_mask(image, mask)
    
    return image, mask

def augment_image_mask(image, mask, flip_prob=0.0, rotate_max_deg=5, brightness_prob=0.5, brightness_range=(0.7, 1.3)):
    """Applies augmentation to both image and mask."""
    
    # Random horizontal flip
    if tf.random.uniform([]) < flip_prob:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Random rotation (using numpy function since TensorFlow lacks arbitrary angle rotation)
    def rotate_image(img, msk):
        angle = np.random.uniform(-rotate_max_deg, rotate_max_deg)
        angle_rad = np.deg2rad(angle)
        img = tf.keras.preprocessing.image.apply_affine_transform(img.numpy(), theta=angle_rad)
        msk = tf.keras.preprocessing.image.apply_affine_transform(msk.numpy(), theta=angle_rad)
        return img, msk

    if tf.random.uniform([]) < 0.5:
        image, mask = tf.numpy_function(rotate_image, [image, mask], [tf.float32, tf.float32])
    
    # Random brightness adjustment (only on image)
    if tf.random.uniform([]) < brightness_prob:
        brightness_factor = tf.random.uniform([], minval=brightness_range[0], maxval=brightness_range[1])
        image = tf.image.adjust_brightness(image, brightness_factor - 1.0)

    return image, mask

def create_dataset(image_paths, mask_paths, batch_size, HEIGHT, WIDTH, augment=False):
    """Creates a tf.data.Dataset from image and mask paths with optional augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(lambda ip, mp: parse_image_mask(ip, mp, HEIGHT, WIDTH, augment),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
