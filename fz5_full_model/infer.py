import os

# File: Path to the saved model
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import sys
import torch
import cv2
from model import UNet

# Path to the saved model
MODEL_PATH = "best_unet_weights.pth"

# Image dimensions (should match model's input size)
HEIGHT, WIDTH = 128, 128

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

def load_model(model_path):
    """
    Load the trained model.
    """
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(image_path, height, width):
    """
    Preprocess the image for the model.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    
    original_shape = image.shape
    resized_image = cv2.resize(image, (width, height))
    tensor = torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor, original_shape

def save_prediction(prediction, original_shape, output_path):
    """
    Save the prediction mask.
    """
    pred_np = prediction.squeeze(0).cpu().numpy()
    pred_resized = cv2.resize(pred_np, (original_shape[1], original_shape[0]))
    pred_binary = (pred_resized > 0.5).astype("uint8") * 255  # Threshold to binary mask
    cv2.imwrite(output_path, pred_binary)
    print(f"Saved mask to {output_path}")

def run_inference(image_path):
    """
    Run inference on a single image.
    """
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print(f"Processing image: {image_path}")
    input_tensor, original_shape = preprocess_image(image_path, HEIGHT, WIDTH)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)

    # Save the output mask
    output_path = f"{os.path.splitext(image_path)[0]}_mask.png"
    save_prediction(probs.squeeze(0), original_shape, output_path)
    print("Inference complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference_single.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    run_inference(image_path)

