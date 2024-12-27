import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from model import UNetBrevitas  # import your quantized U-Net definition

##############################################################################
# Settings
##############################################################################
TEST_DIR = "./test_data"                   # Directory containing test images
MODEL_WEIGHTS = "best_unet_brevitas_weights.pth"  # Path to saved model weights
HEIGHT, WIDTH = 128, 128                  # Model expects 128x128 input
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1) Create the model on CPU first
    model = UNetBrevitas(in_channels=1, out_channels=1)
    
    # 2) Load the state dict onto CPU
    state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)

    # 3) Now move the entire model (including buffers) to DEVICE
    model.to(DEVICE)
    model.eval()

    # 4) Define the transform
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((HEIGHT, WIDTH)),
        T.ToTensor()
    ])

    # 5) Scan test directory
    if not os.path.isdir(TEST_DIR):
        print(f"Test directory '{TEST_DIR}' not found.")
        return

    test_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if os.path.isfile(os.path.join(TEST_DIR, f))
    ])
    if not test_files:
        print(f"No images found in '{TEST_DIR}'.")
        return

    print(f"Found {len(test_files)} test images in '{TEST_DIR}'.")

    # 6) Inference loop
    for file_name in test_files:
        img_path = os.path.join(TEST_DIR, file_name)
        print(f"\nProcessing file: {img_path}")

        # Load and transform image (CPU)
        image_pil = Image.open(img_path)
        input_tensor = transform(image_pil)  # shape [1, H, W]

        # Move the input_tensor to the same device as model
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  # shape [1, 1, H, W]

        with torch.no_grad():
            output_logits = model(input_tensor)  # shape [1, 1, H, W]

        # Convert logits -> prob
        output_prob = torch.sigmoid(output_logits)

        # Move back to CPU for plotting
        input_np = input_tensor.squeeze(0).squeeze(0).cpu().numpy()
        output_np = output_prob.squeeze(0).squeeze(0).cpu().numpy()

        print("Logits:", output_logits.min().item(), output_logits.max().item())
        print("Prob:", output_prob.min().item(), output_prob.max().item())

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(input_np, cmap='gray')
        axs[0].set_title("Input")
        axs[0].axis("off")

        axs[1].imshow(output_np, cmap='gray')
        axs[1].set_title("Output (Sigmoid)")
        axs[1].axis("off")

        plt.suptitle(f"File: {file_name}")
        plt.show()


if __name__ == "__main__":
    main()
