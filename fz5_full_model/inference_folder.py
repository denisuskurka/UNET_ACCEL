# File: fz5_full_model/inference_folder.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import torch
import cv2
from torch.utils.data import DataLoader
from model import UNet
from dataset_infer import RealSegDataset  # Your custom dataset class

# Path to the saved model
MODEL_PATH = "best_unet_weights.pth"

# Directory containing test images
TEST_IMAGES_DIR = "./test_data"

# Directory to save predictions
OUTPUT_DIR = "./predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def save_predictions(predictions, file_names, original_shapes, output_dir):
    """
    Save predictions to the specified output directory.
    """
    for pred, file_name, original_shape in zip(predictions, file_names, original_shapes):
        # Reshape prediction back to original dimensions
        pred_np = pred.squeeze(0).cpu().numpy()
        pred_resized = cv2.resize(pred_np, (original_shape[1], original_shape[0]))
        pred_binary = (pred_resized > 0.5).astype("uint8") * 255  # Threshold to binary mask

        # Save prediction
        output_path = os.path.join(output_dir, f"pred_{file_name}")
        cv2.imwrite(output_path, pred_binary)
        print(f"Saved prediction to {output_path}")

def run_inference(model, test_loader, output_dir):
    """
    Run inference on the test dataset.
    """
    model.eval()
    predictions = []
    file_names = []
    original_shapes = []

    with torch.no_grad():
        for images, file_info in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu())
            file_names.extend([info["file_name"] for info in file_info])
            original_shapes.extend([info["original_shape"] for info in file_info])

    save_predictions(predictions, file_names, original_shapes, output_dir)

def custom_collate_fn(batch):
    """
    Custom collate function to handle metadata in batches.
    """
    images = torch.stack([item[0] for item in batch])  # Stack image tensors
    file_info = [item[1] for item in batch]  # Keep metadata as a list
    return images, file_info

def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Preparing test dataset...")
    # Initialize dataset
    test_dataset = RealSegDataset(
        images_dir=TEST_IMAGES_DIR,  # Path to test images
        masks_dir=None,  # No masks needed for inference
        height=HEIGHT,
        width=WIDTH,
        augment=False  # Disable augmentations for inference
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time during inference
        shuffle=False,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )

    print("Running inference...")
    run_inference(model, test_loader, OUTPUT_DIR)
    print("Inference complete.")


if __name__ == "__main__":
    main()
