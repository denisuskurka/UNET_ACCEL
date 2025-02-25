import os
import time
import tensorflow as tf
import numpy as np
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
from hls4ml.model.profiling import numerical
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pprint
import plotting
from loss import bce_dice_loss, dice_coefficient

from dataset import get_image_mask_paths, create_dataset

# Force TensorFlow to use the CPU only.
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

# Setup custom objects for loading the model
co = {
    "loss":bce_dice_loss(bce_weight=0.3),
    "dice_coefficient":dice_coefficient
}
_add_supported_quantized_objects(co)
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

# Load and strip pruning from the quantized model.
qmodel = tf.keras.models.load_model('quantized_cnn_model_final_128_20.h5', custom_objects=co)
qmodel = strip_pruning(qmodel)

# Then the QKeras model
hls_config_q = hls4ml.utils.config_from_keras_model(qmodel, granularity='model', backend='VivadoAccelerator', board='fz5')
hls_config_q['Model']['ReuseFactor'] = 4
hls_config_q['Model']['Precision'] = 'ap_fixed<32,8>'
hls_config_q['Flows'] = ['vivadoaccelerator:fifo_depth_optimization']
#hls_config_q['Part'] = 'xczu5ev-sfvc784-1-i'
plotting.print_dict(hls_config_q)

hls4ml.model.optimizer.get_optimizer('vivado:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)

cfg_q = hls4ml.converters.create_config(backend='VivadoAccelerator')
cfg_q['IOType'] = 'io_stream'  # Must set this if using CNNs!
cfg_q['HLSConfig'] = hls_config_q
cfg_q['KerasModel'] = qmodel
cfg_q['OutputDir'] = 'quantized_pruned_cnn/'
cfg_q['Board'] = 'fz5'

hls_model_q = hls4ml.converters.keras_to_hls(cfg_q)
hls_model_q.compile()

# Compare the numerical output of the two models.
numerical(model=qmodel, hls_model=hls_model_q)
hls4ml.utils.plot_model(hls_model_q, show_shapes=True, show_precision=True, to_file="model.png")

# ---------------------------
# Synthesize!
# ---------------------------
hls_model_q.build(reset=False, csim=True, synth=True, export=True, cosim=True, bitfile=True)
