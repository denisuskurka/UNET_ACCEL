#!/usr/bin/env python
# File: vitis_ai/ellipse_regressor/eval_graph.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
eval_graph.py - Evaluate Quantized Regression Model

Features:
  - Loads a frozen/quantized graph (.pb).
  - Generates Ground Truth on-the-fly by fitting ellipses to masks (OpenCV).
  - Calculates Mean Absolute Error (MAE) and MSE.
"""

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from progressbar import ProgressBar

# Reduce TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import decent_q to register Quantized Ops (FixNeuron)
try:
    import tensorflow.contrib.decent_q
except ImportError:
    pass

def get_ellipse_gt(mask_path):
    """
    Reads a mask image and calculates the ground truth ellipse parameters.
    Returns: np.array([cx, cy, semi_axis1, semi_axis2, angle])
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(5, dtype=np.float32)

    # Threshold
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    cnts_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            ((cx, cy), (w, h), angle) = cv2.fitEllipse(c)
            # Return params: cx, cy, semi-axis1, semi-axis2, angle
            # Note: fitEllipse returns full diameters (w,h), model predicts semi-axes (radius)
            return np.array([cx, cy, w/2.0, h/2.0, angle], dtype=np.float32)

    # Return zeros if no ellipse found
    return np.zeros(5, dtype=np.float32)

def preprocess_image(path, height, width):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.reshape((height, width, 1))
    return img

def graph_eval(graph_path, input_node, output_node, batchsize, height, width):
    
    # 1. Load Graph
    print(f"[INFO] Loading graph: {graph_path}")
    with tf.io.gfile.GFile(graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    # Import graph into default graph
    tf.import_graph_def(graph_def, name='')

    # 2. Prepare Data Pairs
    images_dir = './data/images'
    masks_dir = './data/masks'
    
    # Match filenames
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    msk_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    common_files = sorted(list(set(img_files).intersection(msk_files)))
    
    if len(common_files) == 0:
        print("[ERROR] No matching image/mask pairs found.")
        return

    print(f"[INFO] Found {len(common_files)} samples for evaluation.")

    # 3. Get Tensors
    graph = tf.compat.v1.get_default_graph()
    try:
        input_tensor = graph.get_tensor_by_name(input_node + ':0')
        output_tensor = graph.get_tensor_by_name(output_node + ':0')
    except KeyError as e:
        print(f"[ERROR] Could not find tensor: {e}")
        print("Hint: Check if input_node/output_node names in 0_setenv.sh match the graph.")
        return

    # 4. Run Evaluation
    total_mae = 0.0
    total_mse = 0.0
    count = 0
    
    with tf.compat.v1.Session() as sess:
        progress = ProgressBar()
        
        # Process in batches
        for i in progress(range(0, len(common_files), batchsize)):
            batch_files = common_files[i : i + batchsize]
            actual_batch_size = len(batch_files)
            
            batch_imgs = []
            batch_gts = []
            
            for fname in batch_files:
                img_path = os.path.join(images_dir, fname)
                mask_path = os.path.join(masks_dir, fname)
                
                # Preprocess Input
                batch_imgs.append(preprocess_image(img_path, height, width))
                # Calculate Ground Truth on-the-fly
                batch_gts.append(get_ellipse_gt(mask_path))

            batch_imgs = np.array(batch_imgs)
            batch_gts = np.array(batch_gts)

            # Run Inference
            preds = sess.run(output_tensor, feed_dict={input_tensor: batch_imgs})
            
            # Calculate Errors
            # MAE = Mean Absolute Error
            abs_diff = np.abs(preds - batch_gts)
            batch_mae = np.mean(abs_diff) # Mean over all params and samples in batch
            
            # MSE = Mean Squared Error
            sq_diff = np.square(preds - batch_gts)
            batch_mse = np.mean(sq_diff)

            # Accumulate weighted average
            total_mae += batch_mae * actual_batch_size
            total_mse += batch_mse * actual_batch_size
            count += actual_batch_size

    avg_mae = total_mae / count
    avg_mse = total_mse / count

    print('\n------------------------------------')
    print(f'Evaluation Results ({count} images)')
    print('------------------------------------')
    print(f'Mean Absolute Error (MAE) : {avg_mae:.4f}')
    print(f'Mean Squared Error (MSE)  : {avg_mse:.4f}')
    print('------------------------------------')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', type=str, required=True, help='Path to quantized .pb')
    ap.add_argument('--input_node', type=str, default='cnn_input')
    ap.add_argument('--output_node', type=str, default='prediction')
    ap.add_argument('-b', '--batchsize', type=int, default=50)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--height', type=int, default=128)
    ap.add_argument('--width', type=int, default=128)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    graph_eval(
        args.graph, 
        args.input_node, 
        args.output_node, 
        args.batchsize, 
        args.height, 
        args.width
    )

if __name__ ==  "__main__":
    main()