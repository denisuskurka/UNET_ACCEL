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
from loss import bce_dice_loss, focal_tversky_loss
from inference import load_and_preprocess_image, show_result, get_image_x_path
from dataset import get_image_mask_paths, create_dataset
from show_raw_result import show_raw_result

def predict_imgs(mdl, output="predicted"):
    for i in range(0, 2):
        image_path = get_image_x_path("./data/stem/images", i)
        if image_path is None:
            print(f"No image files found!.")
        image_hdr = load_and_preprocess_image(image_path)
        image = np.ascontiguousarray(image_hdr)

        # Predict for all images!
        pred = mdl.predict(image)
        pred.tofile("./data/output/"+output+"_"+str(i)+".bin")

# Force TensorFlow to use the CPU only.
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

# Setup custom objects for loading the model
co = {
    "loss":focal_tversky_loss,
}
_add_supported_quantized_objects(co)
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

# Load and strip pruning from the quantized model.
qmodel = tf.keras.models.load_model('stem_model.h5', custom_objects=co)
qmodel = strip_pruning(qmodel)

# Then the QKeras model
hls_config_q = hls4ml.utils.config_from_keras_model(qmodel, granularity='model', backend='VivadoAccelerator')
hls_config_q['Model']['ReuseFactor'] = 2
hls_config_q['Model']['Precision'] = 'ap_fixed<32,8>'
hls_config_q['Flows'] = ['vivadoaccelerator:fifo_depth_optimization']
hls_config_q['Board'] = 'fz5'
hls_config_q['Part'] = 'xczu5ev-sfvc784-1-i'
plotting.print_dict(hls_config_q)

hls4ml.model.optimizer.get_optimizer('vivado:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)

cfg_q = hls4ml.converters.create_config(backend='VivadoAccelerator')
cfg_q['IOType'] = 'io_stream'  # Must set this if using CNNs!
cfg_q['HLSConfig'] = hls_config_q
cfg_q['KerasModel'] = qmodel
cfg_q['OutputDir'] = 'quantized_pruned_cnn/'
cfg_q['AcceleratorConfig']['Board'] = 'fz5'
cfg_q['Board'] = 'fz5'

hls_model_q = hls4ml.converters.keras_to_hls(cfg_q)
hls_model_q.compile()

#predict_imgs(hls_model_q)

# Compare the numerical output of the two models.
numerical(model=qmodel, hls_model=hls_model_q)
hls4ml.utils.plot_model(hls_model_q, show_shapes=True, show_precision=True, to_file="model.png")

# ---------------------------
# Synthesize!
# ---------------------------
hls_model_q.build(reset=False, csim=True, synth=True, export=True, cosim=True, bitfile=False)

predict_imgs(hls_model_q)
