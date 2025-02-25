#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1) Load the raw binary data as float32
    data = np.fromfile("Y_test.bin", dtype=np.float32)
    
    # 2) Reshape to 128×128 (adjust if your image dimensions differ)
    data = data.reshape((128, 128))
    
    # 3) Show the image with a grayscale colormap
    plt.imshow(data, cmap='gray')
    plt.title("Y_test.bin Image")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
