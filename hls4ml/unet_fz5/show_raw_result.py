#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1) Load file as float32 and reshape to 128×128
    data_float = np.fromfile("Y_test.bin", dtype=np.float32)
    data_float = data_float.reshape((128, 128))

    # 2) Threshold at zero -> produce a binary mask
    mask_thresholded = (data_float > 0).astype(np.float32)

    # 3) Display side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: raw logits
    im0 = axes[0].imshow(data_float, cmap='gray')
    axes[0].set_title("Raw Output (float)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Right: thresholded binary mask
    im1 = axes[1].imshow(mask_thresholded, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Thresholded Mask (>0 => 1)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
