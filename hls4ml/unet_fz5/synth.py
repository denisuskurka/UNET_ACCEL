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
qmodel = tf.keras.models.load_model('quantized_cnn_model_final.h5', custom_objects=co)
qmodel = strip_pruning(qmodel)

# Then the QKeras model
hls_config_q = hls4ml.utils.config_from_keras_model(qmodel, granularity='model')
hls_config_q['Model']['ReuseFactor'] = 4
hls_config_q['Model']['Precision'] = 'ap_fixed<32,8>'
#hls_config_q['Model']['Strategy'] = 'Resource'
hls_config_q['Part'] = 'xczu5ev-sfvc784-1-i'
#hls_config_q['Strategy'] = 'Resource'
hls_config_q['Flows'] = ['vivado:fifo_depth_optimization']
hls4ml.model.optimizer.get_optimizer('vivado:fifo_depth_optimization').configure(profiling_fifo_depth=100_000)
plotting.print_dict(hls_config_q)

cfg_q = hls4ml.converters.create_config(backend='Vivado')
cfg_q['IOType'] = 'io_stream'  # Must set this if using CNNs!
cfg_q['HLSConfig'] = hls_config_q
cfg_q['KerasModel'] = qmodel
cfg_q['OutputDir'] = 'quantized_pruned_cnn/'
cfg_q['XilinxPart'] = 'xczu5ev-sfvc784-1-i'
cfg_q['Part'] = 'xczu5ev-sfvc784-1-i'

hls_model_q = hls4ml.converters.keras_to_hls(cfg_q)
hls_model_q.compile()

# Compare the numerical output of the two models.
numerical(model=qmodel, hls_model=hls_model_q)
hls4ml.utils.plot_model(hls_model_q, show_shapes=True, show_precision=True, to_file="model.png")

# ---------------------------
# Synthesize!
# ---------------------------
hls_model_q.build(reset=False, csim=True, synth=True, export=True, cosim=True)

#!sed -n '30,45p' quantized_pruned_cnn/myproject_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper_utilization_placed.rpt

def getReports(indir):
    data_ = {}

    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))

    if report_vsynth.is_file() and report_csynth.is_file():
        print('Found valid vsynth and synth in {}! Fetching numbers'.format(indir))

        # Get the resources from the logic synthesis report
        with report_vsynth.open() as report:
            lines = np.array(report.readlines())
            data_['lut'] = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
            data_['ff'] = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
            data_['bram'] = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
            data_['dsp'] = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
            data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
            data_['ff_rel'] = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
            data_['bram_rel'] = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
            data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])

        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
            data_['latency_clks'] = int(lat_line.split('|')[2])
            data_['latency_mus'] = float(lat_line.split('|')[2]) * 5.0 / 1000.0
            data_['latency_ii'] = int(lat_line.split('|')[6])

    return data_

from pathlib import Path

import pprint

data_quantized_pruned = getReports('quantized_pruned_cnn')

print("\n Resource usage and latency: Pruned + quantized")
pprint.pprint(data_quantized_pruned)
