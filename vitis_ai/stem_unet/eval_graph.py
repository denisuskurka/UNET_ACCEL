import sys

# File: reduce TensorFlow messages in console
# Author: Denis Kurka
# Year: 2025
# License: CC0

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

from progressbar import ProgressBar

# reduce TensorFlow messages in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


from tensorflow.python.platform import gfile
import tensorflow.contrib.decent_q


def list_image_mask_pairs(images_dir, masks_dir):
    image_names = sorted(os.listdir(images_dir))
    mask_names = sorted(os.listdir(masks_dir))
    image_paths = [os.path.join(images_dir, n) for n in image_names]
    mask_paths = [os.path.join(masks_dir, n) for n in mask_names]
    # Optional: enforce matching pairs by filename
    return list(zip(image_paths, mask_paths))

def preprocess_image(path, height, width):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img[..., None]  # Shape: (H, W, 1)
    return img

def preprocess_mask(path, height, width):
    # Same steps as image, but do not scale if mask is already 0/1
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)  # Binarize
    mask = mask[..., None]  # Shape: (H, W, 1)
    return mask

def dice_coef(y_true, y_pred):
    # y_true, y_pred: (H, W, 1)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-6)

def iou_score(y_true, y_pred):
    # Intersection over Union
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(np.maximum(y_true_f, y_pred_f))
    return intersection / (union + 1e-6)

def graph_eval(input_graph_def, graph_path, input_node, output_node, batchsize, height, width):
    input_graph_def.ParseFromString(tf.io.gfile.GFile(graph_path, "rb").read())
    tf.import_graph_def(input_graph_def, name='')

    images_dir = './data/images'
    masks_dir = './data/masks'
    pairs = list_image_mask_pairs(images_dir, masks_dir)
    total_batches = int(np.ceil(len(pairs) / batchsize))

    images_ph = tf.compat.v1.get_default_graph().get_tensor_by_name(input_node + ':0')
    logits = tf.compat.v1.get_default_graph().get_tensor_by_name(output_node + ':0')

    all_dice = []
    all_iou = []

    with tf.compat.v1.Session() as sess:
        progress = ProgressBar()
        for batch_idx in progress(range(total_batches)):
            batch_pairs = pairs[batch_idx*batchsize : (batch_idx+1)*batchsize]
            imgs = [preprocess_image(ipath, height, width) for ipath, _ in batch_pairs]
            masks = [preprocess_mask(mpath, height, width) for _, mpath in batch_pairs]
            imgs = np.stack(imgs, axis=0)
            masks = np.stack(masks, axis=0)
            # Run graph
            pred = sess.run(logits, feed_dict={images_ph: imgs})
            # Postprocess prediction: assume channel last, output in logits or [0,1]
            pred_bin = (pred > 0.5).astype(np.float32)  # threshold if needed (adjust if needed)
            # Calculate metrics per sample in batch
            for i in range(len(batch_pairs)):
                dice = dice_coef(masks[i], pred_bin[i])
                iou = iou_score(masks[i], pred_bin[i])
                all_dice.append(dice)
                all_iou.append(iou)
    
    print('Evaluation completed on {} images.'.format(len(pairs)))
    print('Mean Dice coefficient: {:.4f}'.format(np.mean(all_dice)))
    print('Mean IoU: {:.4f}'.format(np.mean(all_iou)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', type=str, default='./quantize_results/quantize_eval_model.pb')
    ap.add_argument('--input_node', type=str, default='cnn_input')  # Or your UNET input tensor name
    ap.add_argument('--output_node', type=str, default='output_conv/Sigmoid')  # Or your output node
    ap.add_argument('-b', '--batchsize', type=int, default=1)
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=256)
    args = ap.parse_args()

    print('\n------------------------------------')
    print('TensorFlow version : ', tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print(' --graph      : ', args.graph)
    print(' --input_node : ', args.input_node)
    print(' --output_node: ', args.output_node)
    print(' --batchsize  : ', args.batchsize)
    print(' --gpu        : ', args.gpu)
    print(' --height     : ', args.height)
    print(' --width      : ', args.width)
    print('------------------------------------\n')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    input_graph_def = tf.Graph().as_graph_def()
    graph_eval(input_graph_def, args.graph, args.input_node, args.output_node, args.batchsize, args.height, args.width)

if __name__ ==  "__main__":
    main()

