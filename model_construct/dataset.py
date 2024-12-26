import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MyDataset(Dataset):
    def __init__(self, root_dir, height=128, width=128):
        """
        root_dir: path to the data folder, which contains:
            - images/ (e.g. 01.png, 02.png, ...)
            - masks/  (e.g. 01_blue.png, 01_green.png, 01_red.png, ...)
        height, width: desired image/mask size
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.height = height
        self.width = width

        # List all .png in image_dir
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))

        # Define transforms (example)
        self.img_transform = T.Compose([
            T.Grayscale(num_output_channels=1),  # ensure single channel
            T.Resize((self.height, self.width)),
            T.ToTensor(),  # => [C, H, W], range [0,1]
        ])
        self.mask_transform = T.Compose([
            T.Grayscale(num_output_channels=1),  # ensure single channel
            T.Resize((self.height, self.width)),
            T.ToTensor(),  # => [1, H, W], range [0,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1) Load the image
        img_path = self.image_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]  # e.g. "01" from "01.png"
        image = Image.open(img_path)
        image = self.img_transform(image)

        # 2) Find all masks with the same ID (e.g. 01_blue.png, 01_green.png, etc.)
        mask_pattern = os.path.join(self.mask_dir, f"{img_id}_*.png")
        mask_files = sorted(glob.glob(mask_pattern))

        # 3) If no mask found, optionally skip or create an empty mask
        if len(mask_files) == 0:
            # Example: create an all-zero mask
            mask = torch.zeros((1, self.height, self.width), dtype=torch.float32)
        else:
            # Combine all found masks into one
            combined_mask = None
            for mpath in mask_files:
                mimg = Image.open(mpath)
                mimg = self.mask_transform(mimg)
                # Convert to binary (threshold), since mask might have grayscale values
                # e.g. threshold at 0.5
                mimg_bin = (mimg > 0.5).float()

                if combined_mask is None:
                    combined_mask = mimg_bin
                else:
                    # union / logical OR => whichever pixel is 1 in any mask
                    combined_mask = torch.maximum(combined_mask, mimg_bin)

            # If for some reason we never set combined_mask, fallback
            if combined_mask is None:
                combined_mask = torch.zeros((1, self.height, self.width), dtype=torch.float32)
            mask = combined_mask

        # Return image [1,H,W], mask [1,H,W]
        return image, mask
