import torch
import os

from model import UNetBrevitas

###############################################################################
# You can set your desired input shape for FINN here:
###############################################################################
HEIGHT = 128
WIDTH = 128

###############################################################################
# 1) Load trained weights into the model
###############################################################################
def load_trained_model(weights_path, device="cpu"):
    model = UNetBrevitas(in_channels=3, out_channels=1)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # switch to eval mode
    return model

###############################################################################
# 2) Export the model to ONNX for FINN
###############################################################################
def export_model_to_onnx(model, onnx_path, device="cpu", height=128, width=128):
    dummy_input = torch.randn(1, 3, height, width).to(device)
    model.to(device)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True
    )
    print(f"Exported ONNX model to {onnx_path}")

###############################################################################
# 3) Outline: Next steps for FINN
###############################################################################
def next_steps_for_finn(onnx_path):
    """
    Explain the next typical steps you might take with FINN after generating ONNX.
    We won't run them here, but this is the typical workflow.
    """
    print("\n=== Next Steps for FINN ===")
    print(f"1) You have {onnx_path}, a quantized ONNX file from Brevitas.")
    print("2) Install FINN (https://github.com/Xilinx/finn).")
    print("3) Use FINN's 'finn.builder.build_dataflow' or transform scripts:")
    print("   e.g., python -m finn.builder.build_dataflow --model <model.onnx> --output_dir <out>")
    print("4) FINN will parse the ONNX, transform, fold, and generate an RTL or HLS-based accelerator for FPGA.")
    print("5) Some FINN features might not fully support ConvTranspose2d. You may need to modify the model or use an alternative upsampling approach.")
    print("6) Check FINN docs for handling U-Net style architectures and large feature maps.")

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) Load trained weights
    weights_path = "unet_brevitas.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained weights {weights_path} not found. Train first!")
    
    model = load_trained_model(weights_path, device=device)

    # 2) Export to ONNX for FINN
    onnx_path = "unet_brevitas_for_finn.onnx"
    export_model_to_onnx(model, onnx_path, device=device, height=HEIGHT, width=WIDTH)

    # 3) Print recommended next steps
    next_steps_for_finn(onnx_path)
