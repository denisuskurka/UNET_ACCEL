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

from dataset import get_image_mask_paths, create_dataset

def show_masks(keras_preds, hls4ml_preds, num_samples=None):
    """
    Display predicted masks from the Keras and hls4ml models side-by-side.
    
    Parameters:
      keras_preds: NumPy array of shape (N, 128, 128, 1) predicted by the Keras model.
      hls4ml_preds: NumPy array of shape (N, 16384) predicted by the hls4ml model.
      num_samples: (Optional) Number of samples to display. If None, all samples are shown.
    """
    # Determine how many samples to display
    total_samples = keras_preds.shape[0]
    if num_samples is None:
        num_samples = total_samples
    else:
        num_samples = min(num_samples, total_samples)
    
    # Reshape hls4ml predictions to (N, 128, 128, 1)
    hls4ml_preds_reshaped = hls4ml_preds.reshape(-1, 128, 128, 1)
    
    # Create a figure with 2 columns for each sample: one for Keras and one for hls4ml
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        # If only one sample, axes is 1D. Convert it to 2D for consistency.
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(num_samples):
        # Plot the Keras prediction
        axes[i, 0].imshow(np.squeeze(keras_preds[i]), cmap='gray')
        axes[i, 0].set_title(f"Keras Prediction #{i}")
        axes[i, 0].axis('off')
        
        # Plot the hls4ml prediction
        axes[i, 1].imshow(np.squeeze(hls4ml_preds_reshaped[i]), cmap='gray')
        axes[i, 1].set_title(f"hls4ml Prediction #{i}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Force TensorFlow to use the CPU only.
tf.config.set_visible_devices([], 'GPU')
print("Running on CPU only.")

# Setup custom objects for loading the model
co = {}
_add_supported_quantized_objects(co)
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

# Load and strip pruning from the quantized model.
qmodel = tf.keras.models.load_model('quantized_cnn_model_cpu.h5', custom_objects=co)
qmodel = strip_pruning(qmodel)

# Create an hls4ml configuration from the Keras model.
hls_config_q = hls4ml.utils.config_from_keras_model(qmodel, granularity='name', backend='VivadoAccelerator')
hls_config_q['Model']['ReuseFactor'] = 1
hls_config_q['Model']['Precision'] = 'ap_fixed<16,6>'
#hls_config_q['LayerName']['output_softmax']['Strategy'] = 'Stable'
plotting.print_dict(hls_config_q)

cfg_q = hls4ml.converters.create_config(backend='VivadoAccelerator')
cfg_q['IOType'] = 'io_stream'  # Must set this if using CNNs!
cfg_q['HLSConfig'] = hls_config_q
cfg_q['KerasModel'] = qmodel
cfg_q['OutputDir'] = 'quantized_pruned_cnn/'
cfg_q['Board'] = 'pynq-z2'

# Convert the Keras model to an hls4ml model and compile it.
hls_model_q = hls4ml.converters.keras_to_hls(cfg_q)
hls_model_q.compile()

# Compare the numerical output of the two models.
numerical(model=qmodel, hls_model=hls_model_q)
hls4ml.utils.plot_model(hls_model_q, show_shapes=True, show_precision=True, to_file=None)

# ---------------------------
# Data for validation
# ---------------------------
# Directories for your data (make sure these paths are correct)
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"
HEIGHT, WIDTH = 128, 128     # image/mask dimensions
BATCH_SIZE = 4               # adjust as needed

# Get image and mask paths; if counts differ, get only the first min(count) pairs.
image_paths, mask_paths = get_image_mask_paths(IMAGES_DIR, MASKS_DIR)
n_samples = len(image_paths)
split_idx = int(0.7 * n_samples)  # 70% for training, 30% for validation

# We'll use the validation split for testing.
train_image_paths, val_image_paths = image_paths[:split_idx], image_paths[split_idx:]
train_mask_paths, val_mask_paths = mask_paths[:split_idx], mask_paths[split_idx:]

# Create validation dataset.
val_ds = create_dataset(val_image_paths, val_mask_paths, BATCH_SIZE, HEIGHT, WIDTH)

# Convert the tf.data.Dataset to NumPy arrays.
val_x_list = []
val_y_list = []
for images, masks in val_ds:
    val_x_list.append(images.numpy())
    val_y_list.append(masks.numpy())
    
val_x = np.concatenate(val_x_list, axis=0)
val_y = np.concatenate(val_y_list, axis=0)

# Optionally, ensure the input is stored in a contiguous array.
val_x = np.ascontiguousarray(val_x)

# ---------------------------
# Test predictions
# ---------------------------
#print("Running predictions on CPU...")
#y_predict_q = qmodel.predict(val_x)
#y_predict_hls4ml_q = hls_model_q.predict(val_x)

# Optionally, print shapes or other diagnostics.
#print("Keras model prediction shape:", y_predict_q.shape)
#print("hls4ml model prediction shape:", y_predict_hls4ml_q.shape)

#show_masks(y_predict_q, y_predict_hls4ml_q, num_samples=5)

# ---------------------------
# Synthesize!
# ---------------------------
hls_model_q.build(csim=False, export=True, bitfile=True)

#!sed -n '30,45p' quantized_pruned_cnn/myproject_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper_utilization_placed.rpt
