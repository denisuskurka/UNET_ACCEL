# File: fz5_full_model/testbench.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import os
import time
import torch
import cv2
from model import UNet

# Path to the saved model
MODEL_PATH = "best_unet_weights.pth"

# Directory containing test images
TEST_IMAGES_DIR = "./test_data"

# Create a unique output directory based on the current timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f"./predictions_{timestamp}"
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

def save_prediction(prediction, original_shape, output_path):
    """
    Save the prediction mask.
    """
    # Use only the first channel of the prediction
    pred_np = prediction[0].squeeze(0).cpu().numpy()  # Convert to NumPy array
    pred_resized = cv2.resize(pred_np, (original_shape[1], original_shape[0]))
    pred_binary = (pred_resized > 0.5).astype("uint8") * 255  # Threshold to binary mask

    # Ensure the output is 2D and has proper type for OpenCV
    pred_binary = pred_binary.astype("uint8")

    # Save the binary mask
    success = cv2.imwrite(output_path, pred_binary)
    if not success:
        raise ValueError(f"Failed to save the mask to {output_path}")

    print(f"Saved mask to {output_path}")


def run_inference(model, image_path):
    """
    Run inference on a single image.
    """
    # Load and preprocess the image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_shape = original_image.shape
    resized_image = cv2.resize(original_image, (WIDTH, HEIGHT))
    input_tensor = torch.from_numpy(resized_image).float().unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)

    return probs, original_shape

def testbench(model, test_images_dir, output_dir):
    """
    Testbench to process all images in a folder and measure performance.
    """
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in the directory: {test_images_dir}")
        return

    total_time = 0
    num_images = len(image_files)

    print(f"Processing {num_images} images...")

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(test_images_dir, image_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_mask.png")

        # Measure time for each image
        start_time = time.time()

        probs, original_shape = run_inference(model, image_path)
        save_prediction(probs, original_shape, output_path)

        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        print(f"[{idx + 1}/{num_images}] Processed {image_file} in {elapsed_time:.4f} seconds.")

    avg_time = total_time / num_images
    fps = num_images / total_time  # Theoretical FPS

    print("\n=== Testbench Results ===")
    print(f"Total images processed: {num_images}")
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time:.4f} seconds")
    print(f"Theoretical FPS: {fps:.2f} frames/second")

def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Running testbench...")
    testbench(model, TEST_IMAGES_DIR, OUTPUT_DIR)
    print("Testbench complete.")

if __name__ == "__main__":
    main()
