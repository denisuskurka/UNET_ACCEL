#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

def show_raw_result(
    path,
    only_left=False,
    save=False,
    outdir=None,
    filename="result.png"
):
    """
    Load a 128×128 float32 binary file, threshold it at zero, and:
      - Show only the left (raw) image if only_left=True
      - Otherwise, show side-by-side raw image (left) and thresholded mask (right)
      - Optionally save to disk if save=True and outdir is provided

    :param path: Path to the .bin file
    :param only_left: If True, only display the raw image
    :param save: If True, save the resulting figure
    :param outdir: Directory to save the figure (used only if save=True)
    :param filename: Name for the saved figure file
    """

    # 1) Load file as float32 and reshape to 128×128
    data_float = np.fromfile(path, dtype=np.float32)
    data_float = data_float.reshape((128, 128))

    # 2) Threshold at zero -> produce a binary mask
    mask_thresholded = (data_float > 0).astype(np.float32)

    # 3) Create figure
    if only_left:
        # Only display the raw image
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(data_float, cmap='gray')
        ax.set_title("Raw Output (float)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    else:
        # Display side by side
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

    # 4) Save or Show
    if save:
        if not outdir:
            raise ValueError("`outdir` must be specified when save=True.")
        os.makedirs(outdir, exist_ok=True)
        save_path = os.path.join(outdir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close(fig)  # Avoid displaying when saving
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage:
    # show_raw_result("./data_stem_input.bin", only_left=True)
    # show_raw_result("./data_stem_input.bin", only_left=False, save=True, outdir="outputs")
    show_raw_result("./data_stem_input.bin", only_left=False, save=True, outdir="outputs")
