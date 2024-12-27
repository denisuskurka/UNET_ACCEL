import torch
import onnx
import os
from finn.util.test import get_test_model_trained
from finn.util.visualization import showSrc, showInNetron
from finn.util.basic import make_build_dir
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.fold_constants import FoldConstants
from finn.util.pytorch import ToTensor
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.core.datatype import DataType
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
import finn.transformation.streamline.absorb as absorb

from model import UNetBrevitas  # Import your model definition

##############################################################################
# Settings
##############################################################################
# Ensure the FINN_BUILD_DIR environment variable is set
if "FINN_BUILD_DIR" not in os.environ:
    raise EnvironmentError("Please set the FINN_BUILD_DIR environment variable.")

# Define build directory from environment variable
build_dir = os.environ["FINN_BUILD_DIR"]  # Typically /workspace inside Docker

# Path to saved model weights (located inside the workspace)
MODEL_WEIGHTS = os.path.join(build_dir, "best_unet_brevitas_weights.pth")  # /workspace/best_unet_brevitas_weights.pth

# Image & mask size
HEIGHT, WIDTH = 160, 160

##############################################################################
# 1) Instantiate the Model
##############################################################################
# Initialize the model architecture
model = UNetBrevitas(in_channels=1, out_channels=1)

# Load the state dictionary into the model
if not os.path.isfile(MODEL_WEIGHTS):
    raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS}")

state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

##############################################################################
# 2) Export to QONNX
##############################################################################
# Define the path for the exported ONNX model
export_onnx_path = os.path.join(build_dir, "unet_finn.onnx")  # /workspace/unet_finn.onnx

# Create the build directory if it doesn't exist
os.makedirs(build_dir, exist_ok=True)

# Generate a dummy input tensor with the correct shape
dummy_input = torch.randn(1, 1, HEIGHT, WIDTH)

try:
    # Export the model to QONNX format
    export_qonnx(model, dummy_input, export_onnx_path)
    print(f"Model successfully exported to {export_onnx_path}")
except Exception as e:
    print("An error occurred during ONNX export:")
    print(e)
    exit(1)

##############################################################################
# 3) Clean Up the ONNX Model
##############################################################################
try:
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    print(f"ONNX model cleaned up and saved to {export_onnx_path}")
except Exception as e:
    print("An error occurred during ONNX cleanup:")
    print(e)
    exit(1)

##############################################################################
# 4) Convert to FINN
##############################################################################
# Initialize ModelWrapper with the exported ONNX model
model = ModelWrapper(export_onnx_path)  # /workspace/unet_finn.onnx

# Apply the transformation to convert QONNX to FINN
model = model.transform(ConvertQONNXtoFINN())

# Define the path for the FINN-converted model
finn_model_path = os.path.join(build_dir, "unet_finn_finn.onnx")  # /workspace/unet_finn_finn.onnx

# Save the FINN-converted model
model.save(finn_model_path)
print(f"FINN-converted model saved to {finn_model_path}")

##############################################################################
# 5) Perform Network Transformations
##############################################################################
# Apply a series of transformations to tidy up the model
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())

# Define the path for the tidied model
tidy_model_path = os.path.join(build_dir, "unet_finn_tidy.onnx")  # /workspace/unet_finn_tidy.onnx

# Save the tidied model
model.save(tidy_model_path)
print(f"Tidied model saved to {tidy_model_path}")

##############################################################################
# 6) Preprocessing Inclusion
##############################################################################
# Include torchvision.transforms.ToTensor() equivalent using finn.util.pytorch.ToTensor
model = ModelWrapper(tidy_model_path)

# Get the global input tensor name and its shape
global_inp_name = model.graph.input[0].name
ishape = model.get_tensor_shape(global_inp_name)

# Create the ToTensor preprocessing model
totensor_pyt = ToTensor()

# Define the path for the preprocessing model
chkpt_preproc_name = os.path.join(build_dir, "unet_finn_preproc.onnx")  # /workspace/unet_finn_preproc.onnx

# Export the preprocessing model
export_qonnx(totensor_pyt, torch.randn(ishape), chkpt_preproc_name)
qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)
print(f"Preprocessing model exported and cleaned up at {chkpt_preproc_name}")

# Initialize ModelWrapper with the preprocessing model
pre_model = ModelWrapper(chkpt_preproc_name)

# Convert the preprocessing model to FINN
pre_model = pre_model.transform(ConvertQONNXtoFINN())

# Merge the preprocessing model with the core model
model = model.transform(MergeONNXModels(pre_model))

# Add input quantization annotation: UINT8 for all BNN-PYNQ models
global_inp_name = model.graph.input[0].name
model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

# Define the path for the model with preprocessing
with_preproc_model_path = os.path.join(build_dir, "unet_finn_with_preproc.onnx")  # /workspace/unet_finn_with_preproc.onnx

# Save the merged model
model.save(with_preproc_model_path)
print(f"Model with preprocessing saved to {with_preproc_model_path}")

##############################################################################
# 7) Postprocessing Inclusion
##############################################################################
# TODO: Define postprocessing steps if necessary (e.g., scaling outputs)

##############################################################################
# 8) Streamlining
##############################################################################
# Initialize ModelWrapper with the model including preprocessing
model = ModelWrapper(with_preproc_model_path)  # /workspace/unet_finn_with_preproc.onnx

# Move initial Mul (from preproc) past the Reshape
model = model.transform(MoveScalarLinearPastInvariants())

# Streamline the model
model = model.transform(Streamline())

# Define the path for the streamlined model
streamlined_model_path = os.path.join(build_dir, "unet_finn_streamlined.onnx")  # /workspace/unet_finn_streamlined.onnx

# Save the streamlined model
model.save(streamlined_model_path)
print(f"Streamlined model saved to {streamlined_model_path}")

##############################################################################
# 9) (Optional) Final Cleanup and Validation
##############################################################################
try:
    # Load and validate the streamlined ONNX model
    onnx_model = onnx.load(streamlined_model_path)
    onnx.checker.check_model(onnx_model)
    print("Streamlined ONNX model is valid and ready for use with FINN.")
except onnx.checker.ValidationError as e:
    print("ONNX model validation failed:")
    print(e)
    exit(1)
except Exception as e:
    print("An unexpected error occurred during ONNX validation:")
    print(e)
    exit(1)
