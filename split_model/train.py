import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from dataset import RealSegDataset  # your custom dataset that loads images/masks
from split_unet import UNetSplit  # <--- updated file from above

###############################################################################
# 1) Hyperparameters & Settings
###############################################################################
HEIGHT, WIDTH = 128, 128  # image & mask size (should match model input size)
BATCH_SIZE = 40
NUM_EPOCHS = 1000
LEARNING_RATE = 0.1
VAL_SPLIT = 0.3
EARLY_STOP_PATIENCE = 100
SAVE_BEST_MODEL_PATH = "best_unet_weights.pth"

# Paths to your data
IMAGES_DIR = "./data/images"
MASKS_DIR = "./data/masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs("progress", exist_ok=True)

###############################################################################
# Visual Inspection
###############################################################################
def save_visual_inspection(
    model, 
    epoch, 
    device, 
    dataset, 
    sample_idx=0,
    out_file="visual.png"
):
    model.eval()

    img, mask = dataset[sample_idx]
    input_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)

    input_np = input_tensor.squeeze(0).squeeze(0).cpu().numpy()
    mask_np = mask.squeeze(0).cpu().numpy()
    prob_np = probs.squeeze(0).squeeze(0).cpu().numpy()

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

    out_path = f"progress/{epoch}_{out_file}"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

###############################################################################
# Dice Loss
###############################################################################
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() + self.eps
        dice_score = 2.0 * intersection / union
        return 1.0 - dice_score

###############################################################################
# Combined BCE + Dice
###############################################################################
class BCEDiceLoss(nn.Module):
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
# Training & Validation
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

def main():
    shutil.rmtree("./progress", ignore_errors=True)
    os.makedirs("./progress", exist_ok=True)

    dataset = RealSegDataset(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        height=HEIGHT,
        width=WIDTH,
        augment=True,
        flip_prob=0.5
    )
    print(f"Total samples: {len(dataset)}")

    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetSplit(
        in_ch=1,
        out_ch=1,
        encoder_channels=[64,128,256,512],
        decoder_channels=[512,256,128,64],
        bottleneck_channels=1024,
        weight_bit_width=8,
        act_bit_width=8
    ).to(DEVICE)

    loss_fn = BCEDiceLoss(bce_weight=0.3)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    model_saved = False

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epoch % 10 == 0 and epoch < 60:
            save_visual_inspection(model, epoch, DEVICE, dataset, sample_idx=0)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if epoch > 60:
                model_saved = True
                torch.save(model.state_dict(), SAVE_BEST_MODEL_PATH)
                print("  * New best model saved.")
                save_visual_inspection(model, epoch, DEVICE, dataset, sample_idx=0)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print("Early stopping due to no improvement.")
                break

    if not model_saved:
        torch.save(model.state_dict(), SAVE_BEST_MODEL_PATH)
        print("Saved LAST model!.")
    print("Training complete.")
    print(f"Best model saved at {SAVE_BEST_MODEL_PATH} with val_loss {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
