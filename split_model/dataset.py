import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class RealSegDataset(Dataset):
    """
    A dataset that:
      - Scans `images_dir` and `masks_dir` for matching filenames.
      - Loads image & mask as grayscale [1, H, W].
      - Resizes to (height, width).
      - Optionally applies data augmentation (random flips, rotation, brightness changes).
        * Horizontal flip / vertical flip for both image & mask.
        * Random rotation for both image & mask.
        * Random brightness ONLY for the image.
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        height=128,
        width=128,
        height_mask=128,
        width_mask=128,
        augment=False,        # enable/disable all augmentation
        flip_prob=0.5,        # probability of flips (horizontal/vertical)
        rotate_prob=0,      # probability of random rotation
        max_rotate_deg=5,    # +/- max degrees for rotation
        brightness_prob=0.5,  # probability of random brightness change
        brightness_range=(0.7, 1.3)  # min/max for brightness factor
    ):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.height = height
        self.width = width
        self.height_mask = height_mask
        self.width_mask = width_mask
        self.augment = augment

        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotate_deg = max_rotate_deg
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range

        # Gather all possible mask filenames
        all_mask_files = sorted(os.listdir(self.masks_dir))

        # Build a list of (image_path, mask_path) pairs
        self.data_pairs = []
        for mask_name in all_mask_files:
            mask_path = os.path.join(self.masks_dir, mask_name)
            image_path = os.path.join(self.images_dir, mask_name)
            if os.path.isfile(image_path):
                self.data_pairs.append((image_path, mask_path))

        # Base transforms (grayscale -> resize)
        # We'll convert to tensor manually in __getitem__ after augmentation
        self.image_transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((self.height, self.width)),
        ])
        self.mask_transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((self.height_mask, self.width_mask)),
        ])

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.data_pairs[idx]
        
        # Load PIL images
        image_pil = Image.open(image_path)
        mask_pil = Image.open(mask_path)

        # Apply the base transforms (grayscale + resize)
        image = self.image_transform(image_pil)  # still PIL after transform
        mask = self.mask_transform(mask_pil)     # still PIL

        # Convert to PIL to keep them consistent for transforms.functional
        # (Note: T.Resize returns a PIL Image if input is PIL, so we're good.)

        # -----------
        # Data Augmentation
        # -----------
        if self.augment:
            # 1) Horizontal flip
            if random.random() < self.flip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            ## 2) Vertical flip
            #if random.random() < self.flip_prob:
            #    image = TF.vflip(image)
            #    mask = TF.vflip(mask)

            # 3) Random rotation
            if random.random() < self.rotate_prob:
                angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # 4) Random brightness (only on image)
            if random.random() < self.brightness_prob:
                brightness_factor = random.uniform(*self.brightness_range)
                image = TF.adjust_brightness(image, brightness_factor)
                # Do not apply brightness to the mask

        # Convert final PIL images to torch.Tensor
        image_tensor = T.ToTensor()(image)  # shape [1, H, W], float in [0,1]
        mask_tensor = T.ToTensor()(mask)    # shape [1, H, W], float in [0,1]

        # Optionally binarize the mask if needed:
        # mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor
