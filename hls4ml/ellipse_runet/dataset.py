# File: hls4ml/ellipse_runet/dataset.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_image_mask_paths(images_dir, masks_dir):
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


def mask_to_ellipse_params(mask):
    mask_np = (mask.numpy() * 255).astype(np.uint8)
    mask_np = np.squeeze(mask_np)

    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        params = [0.0, 0.0, 1.0, 1.0, 0.0]
    else:
        cnt = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (axis1, axis2), angle = ellipse
        params = [cx, cy, axis1 / 2, axis2 / 2, angle]

    return tf.convert_to_tensor(params, dtype=tf.float32)


def parse_image_mask(image_path, mask_path, HEIGHT, WIDTH):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [HEIGHT, WIDTH])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, [HEIGHT, WIDTH])

    ellipse_params = tf.py_function(mask_to_ellipse_params, [mask], tf.float32)
    ellipse_params.set_shape([5])

    return image, ellipse_params


def create_dataset(image_paths, mask_paths, batch_size, HEIGHT, WIDTH):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(lambda ip, mp: parse_image_mask(ip, mp, HEIGHT, WIDTH),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def draw_ellipse_on_blank(HEIGHT, WIDTH, params):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cx, cy, axis1, axis2, angle = params
    center = (int(cx), int(cy))
    axes = (int(axis1), int(axis2))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=-1)
    return mask


def main():
    images_dir = "data/images"
    masks_dir = "data/masks"
    HEIGHT, WIDTH = 128, 128
    batch_size = 1

    image_paths, mask_paths = get_image_mask_paths(images_dir, masks_dir)
    dataset = create_dataset(image_paths, mask_paths, batch_size, HEIGHT, WIDTH)

    for image, params in dataset.take(5):
        img = image[0].numpy().squeeze()
        p = params[0].numpy()
        print(f"Ellipse Params: {p}")

        reconstructed_mask = draw_ellipse_on_blank(HEIGHT, WIDTH, p)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Image')

        original_mask = draw_ellipse_on_blank(HEIGHT, WIDTH, p)  # optional: use gt mask here
        plt.subplot(1, 3, 2)
        plt.imshow(original_mask, cmap='gray')
        plt.title('Ellipse From Params')

        plt.subplot(1, 3, 3)
        plt.imshow(img, cmap='gray')
        plt.imshow(reconstructed_mask, cmap='jet', alpha=0.5)
        plt.title('Overlay')

        plt.show()


if __name__ == "__main__":
    main()
