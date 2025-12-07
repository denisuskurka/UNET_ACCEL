'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 ...
'''

import os
import argparse

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# CRITICAL PRE-SETUP
# ==============================================================================
import tensorflow as tf

try:
    # Disable Eager Execution to allow access to the Graph and Session
    tf.compat.v1.disable_eager_execution()
    print("[INFO] Eager Execution Disabled successfully.")
except Exception as e:
    print(f"[ERROR] Could not disable eager execution: {e}")

# Import other modules AFTER disabling eager execution
from loss import bce_dice_loss, focal_tversky_loss
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json, load_model
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.compat.v1 import graph_util

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
        custom_objects = {}
        _add_supported_quantized_objects(custom_objects)
        custom_objects['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
        
        # Add custom losses
        custom_objects['bce_dice_loss'] = bce_dice_loss
        custom_objects['focal_tversky_loss'] = focal_tversky_loss
        
        loaded_model = load_model(keras_hdf5, custom_objects=custom_objects, compile=False)


    ##############################################
    # Freeze Graph
    ##############################################

    print ('Keras model information:')
    print (' Input names :', loaded_model.inputs)
    print (' Output names:', loaded_model.outputs)
    print('-------------------------------------')

    # Fetch the tensorflow session using the Keras backend
    sess = tf.compat.v1.keras.backend.get_session()

    print('-------------------------------------')
    print(' FREEZING GRAPH INTERNALLY...')
    
    # ==========================================================================
    # FIX: Add Identity Nodes to guarantee clean names
    # ==========================================================================
    output_names = []
    
    # Iterate over the Keras Output Tensors
    for i, output_tensor in enumerate(loaded_model.outputs):
        # We create a new name for the output node
        # e.g., "frozen_output_0", "frozen_output_1"
        node_name = "frozen_output_" + str(i)
        
        # We attach a TensorFlow Identity Op to the Keras output tensor.
        # This effectively adds a new node to the graph that simply passes the data through.
        # This ensures the node name exists and is exactly what we expect.
        tf.identity(output_tensor, name=node_name)
        
        output_names.append(node_name)

    print(' Generated clean output node names:', output_names)

    # Freeze the graph using the session we already have
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_names
    )
    
    # Define the frozen graph path
    chkpt_dir = os.path.dirname(tf_ckpt)
    
    # Ensure directory exists
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    frozen_graph_path = os.path.join(chkpt_dir, 'frozen_graph.pb')
    
    # Write the frozen graph to disk
    tf.io.write_graph(frozen_graph_def, chkpt_dir, 'frozen_graph.pb', as_text=False)
    
    print(' Frozen graph created :', frozen_graph_path)
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
