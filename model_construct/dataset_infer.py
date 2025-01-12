import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2

class RealSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, height=128, width=128, augment=False, **kwargs):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.height = height
        self.width = width
        self.augment = augment

        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        if masks_dir:
            self.mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.mask_paths = None  # No masks required for inference

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        original_shape = img.shape  # Save original dimensions

        # Resize and normalize the image
        img_resized = cv2.resize(img, (self.width, self.height))
        img_normalized = img_resized / 255.0
        img_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0)  # Shape: [1, H, W]

        file_info = {
            "file_name": os.path.basename(img_path),
            "original_shape": original_shape
        }

        return img_tensor, file_info
