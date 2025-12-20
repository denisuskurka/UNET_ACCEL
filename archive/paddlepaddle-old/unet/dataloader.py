# dataloader.py
import paddle
from paddle.io import Dataset
import paddle.vision.transforms as T
import numpy as np
from model import IMG_HEIGHT, IMG_WIDTH  # Import the height and width

class SyntheticDataset(Dataset):
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_samples=100):
        self.img_height = img_height
        self.img_width = img_width
        self.num_samples = num_samples
        self.transform = T.Compose([T.Normalize(mean=[0.5], std=[0.5])])

    def __getitem__(self, index):
        # Single-channel image, then expand to 3 channels
        image = np.random.rand(1, self.img_height, self.img_width).astype('float32')
        image = np.repeat(image, 3, axis=0)  # Expand to (3, height, width)
        image = self.transform(image)

        # Create mask as integer class labels (0 or 1)
        mask = (np.random.rand(self.img_height, self.img_width) > 0.5).astype('int64')  # Example binary mask

        return paddle.to_tensor(image), paddle.to_tensor(mask)

    def __len__(self):
        return self.num_samples
