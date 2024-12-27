import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from dataset import RealSegDataset  # your custom dataset that loads images/masks
from model import UNetBrevitas       # your quantized U-Net definition

###############################################################################
# 1) Hyperparameters & Settings
###############################################################################
HEIGHT, WIDTH = 160, 160   # image & mask size
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 0.0005
VAL_SPLIT = 0.2
EARLY_STOP_PATIENCE = 8
SAVE_BEST_MODEL_PATH = "best_unet_brevitas_weights.pth"

# Paths to your data
IMAGES_DIR = "/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct/data/images"
MASKS_DIR  = "/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct/data/masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Make sure progress/ exists
os.makedirs("progress", exist_ok=True)

def save_visual_inspection(
    model, 
    epoch, 
    device, 
    dataset, 
    sample_idx=0,
    out_file="visual.png"
):
    """
    Runs inference on `dataset[sample_idx]`, plots:
      (1) input image
      (2) ground truth mask
      (3) predicted probability map
    Then saves to `out_file` in progress/ folder.
    Overwrites with <epoch> as prefix so you get a series of images over epochs.
    """
    model.eval()

    # Grab a single sample from the dataset
    img, mask = dataset[sample_idx]  # each shape [1,H,W]
    input_tensor = img.unsqueeze(0).to(device)  # [1,1,H,W]

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)   # [1,1,H,W]
        probs = torch.sigmoid(logits)  # [1,1,H,W]

    # Convert to CPU numpy for plotting
    input_np = input_tensor.squeeze(0).squeeze(0).cpu().numpy()
    mask_np = mask.squeeze(0).cpu().numpy()
    prob_np = probs.squeeze(0).squeeze(0).cpu().numpy()

    # Plot side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input_np, cmap='gray')
    axs[0].set_title("Input")
    axs[0].axis("off")

    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(prob_np, cmap='gray')
    axs[2].set_title("Prediction (Sigmoid)")
    axs[2].axis("off")

    fig.suptitle(f"Epoch {epoch} | Sample {sample_idx}")
    plt.tight_layout()

    # Save & close
    out_path = f"progress/{epoch}_{out_file}"  # e.g. progress/1_visual.png
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

###############################################################################
# Dice Loss
###############################################################################
class DiceLoss(nn.Module):
    """
    Soft Dice Loss. Encourages the model to handle small foreground regions better.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # logits: [N, 1, H, W] (raw output from model)
        # targets: [N, 1, H, W] in [0,1]
        probs = torch.sigmoid(logits)  # [N,1,H,W], in [0..1]
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() + self.eps
        dice_score = 2.0 * intersection / union
        return 1.0 - dice_score  # 1 - DSC => 0 is perfect overlap

###############################################################################
# Combined BCE + Dice
###############################################################################
class BCEDiceLoss(nn.Module):
    """
    Weighted combination of BCEWithLogitsLoss and DiceLoss.
    Helps with class imbalance.
    """
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        bce_val = self.bce_loss(logits, targets)
        dice_val = self.dice_loss(logits, targets)
        return self.bce_weight * bce_val + (1.0 - self.bce_weight) * dice_val

###############################################################################
# Training & Validation Loops
###############################################################################
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(loader)

###############################################################################
# Main
###############################################################################
def main():
    # Create dataset
    dataset = RealSegDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        height=HEIGHT,
        width=WIDTH,
        augment=True,        
        flip_prob=0.5,       
        rotate_prob=0.5,     
        max_rotate_deg=15,   
        brightness_prob=0.5, 
        brightness_range=(0.7, 1.3)
    )
    print(f"Total samples: {len(dataset)}")

    # Inspect first sample
    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f"First sample shape: img {img.shape}, mask {mask.shape}")
        fg_ratio = (mask > 0.5).float().mean().item()
        print(f"Mask foreground ratio: {fg_ratio:.3f}")

    # Train/Val split
    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)

    # Create model, loss, optimizer
    model = UNetBrevitas(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    # Early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # CSV logging: create or overwrite
    csv_path = "progress/loss_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train & validate
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss   = validate_one_epoch(model, val_loader, loss_fn, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        # Visual inspection
        save_visual_inspection(
            model=model,
            epoch=epoch,
            device=DEVICE,
            dataset=dataset,   # or val_subset.dataset if you'd rather see a validation example
            sample_idx=0,
            out_file="visual.png"
        )

        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_BEST_MODEL_PATH)
            print("  * New best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print("Early stopping due to no improvement.")
                break

    print("Training complete.")
    print(f"Best model saved at {SAVE_BEST_MODEL_PATH} with val_loss {best_val_loss:.4f}")

    # Plot final train/val loss curves
    epochs_range = range(1, len(train_losses)+1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("progress/loss_plot.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
