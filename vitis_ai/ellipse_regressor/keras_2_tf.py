'''

# File: 
# Author: Denis Kurka
# Year: 2025
# License: CC0

'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 ...
'''

import os
import argparse

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# ==============================================================================
# CRITICAL: Disable Eager Execution for TF 1.x / Vitis AI compatibility
# ==============================================================================
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.python.framework import graph_util

def keras_convert(keras_json, keras_hdf5, tf_ckpt):

    ##############################################
    # load the saved Keras model
    ##############################################

    # set learning phase for no training
    backend.set_learning_phase(0)

    # Load Model
    if (keras_json != ''):
        json_file = open(keras_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(keras_hdf5)
    else:
        # We use compile=False to avoid loading custom loss functions 
        # (which aren't needed for freezing)
        loaded_model = load_model(keras_hdf5, compile=False)


    ##############################################
    # Freeze Graph Internally (No Checkpoint needed)
    ##############################################

    print ('Keras model information:')
    print (' Input names :', loaded_model.inputs)
    print (' Output names:', loaded_model.outputs)
    print('-------------------------------------')

    # Fetch the tensorflow session using the Keras backend
    sess = tf.compat.v1.keras.backend.get_session()

    print('-------------------------------------')
    print(' FREEZING GRAPH INTERNALLY...')
    
    # 1. Rename the Output Node
    # The regression model output is often named something messy like 'dense_1/BiasAdd'.
    # We add an Identity node named 'prediction' to force a clean name.
    
    output_tensor = loaded_model.outputs[0]
    # This creates a new node in the graph called "prediction"
    final_tensor = tf.identity(output_tensor, name="prediction")
    
    # 2. Define Output Node Names
    output_node_names = ["prediction"]
    
    # 3. Freeze the graph (Convert variables to constants)
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names
    )
    
    # 4. Save the .pb file
    # We derive the path from the tf_ckpt argument to keep compatibility with your shell script
    # If tf_ckpt is "./build/tf_chkpt/tf_float.ckpt", we save "./build/tf_chkpt/frozen_graph.pb"
    chkpt_dir = os.path.dirname(tf_ckpt)
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    frozen_graph_path = os.path.join(chkpt_dir, 'frozen_graph.pb')
    
    with tf.io.gfile.GFile(frozen_graph_path, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
    
    print(' Frozen graph created :', frozen_graph_path)
    print(' Final Input Node     :', loaded_model.inputs[0].name.split(':')[0])
    print(' Final Output Node    :', output_node_names)
    print('-------------------------------------')

    return

def run_main():

    # command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-kj', '--keras_json',
                    type=str,
                    default='',
                    help='path of Keras JSON. Default is empty string to indicate no JSON file')
    ap.add_argument('-kh', '--keras_hdf5',
                    type=str,
                    default='./model.hdf5',
                    help='path of Keras HDF5. Default is ./model.hdf5')
    ap.add_argument('-tf', '--tf_ckpt',
                    type=str,
                    default='./tf_float.ckpt',
                    help='path of TensorFlow checkpoint. Default is ./tf_float.ckpt')           
    args = ap.parse_args()

    print('-------------------------------------')
    print('keras_2_tf command line arguments:')
    print(' --keras_json:', args.keras_json)
    print(' --keras_hdf5:', args.keras_hdf5)
    print(' --tf_ckpt   :', args.tf_ckpt)
    print('-------------------------------------')

    keras_convert(args.keras_json, args.keras_hdf5, args.tf_ckpt)

if __name__ == '__main__':
    run_main()

