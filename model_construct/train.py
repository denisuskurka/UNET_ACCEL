import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import UNetBrevitas

###############################################################################
# Global shape parameters for training & exporting
###############################################################################
HEIGHT = 128
WIDTH = 128

###############################################################################
# Synthetic Dataset (random data)
###############################################################################
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image: [3, HEIGHT, WIDTH]
        image = torch.randn(3, HEIGHT, WIDTH)
        # Random mask: [1, HEIGHT, WIDTH]
        mask = (torch.randn(1, HEIGHT, WIDTH) > 0).float()
        return image, mask

###############################################################################
# Training function
###############################################################################
def train_unet_brevitas(
    model,
    data_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs=5,
    log_interval=10
):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(data_loader)}], "
                      f"Loss: {loss.item():.4f}")

###############################################################################
# Main script
###############################################################################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Create random dataset
    train_dataset = SyntheticDataset(num_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Create model
    model = UNetBrevitas(in_channels=3, out_channels=1).to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_unet_brevitas(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        device=device,
        num_epochs=1,
        log_interval=10
    )

    print("Training complete!")

    # ------------------------------------------------------------------------
    # 1) Save the model weights (PyTorch format)
    # ------------------------------------------------------------------------
    model_save_path = "unet_brevitas.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

    # ------------------------------------------------------------------------
    # 2) Export to ONNX
    #    We use the same (HEIGHT, WIDTH) for the input shape.
    # ------------------------------------------------------------------------
    onnx_export_path = "unet_brevitas.onnx"

    # We need a dummy input of the correct shape
    dummy_input = torch.randn(1, 3, HEIGHT, WIDTH).to(device)

    # Switch model to eval mode
    model.eval()

    # Export
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_export_path,
        opset_version=11,      # FINN typically expects opset 11 or newer
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True
    )
    print(f"ONNX model saved to {onnx_export_path}")
