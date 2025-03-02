#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def encode(xi):
    return np.int32(round(xi * 2**24))  # 24 fractional bits
def decode(yi):
    return yi * 2**-24
encode_v = np.vectorize(encode)  # unused in this script
decode_v = np.vectorize(decode)

def main():
    # 1) Read raw binary data as int32 (since we encoded with encode()->int32).
    raw_data = np.fromfile("Y_test.bin", dtype=np.int32)

    # 2) Decode from fixed-point => float.
    data_float = decode_v(raw_data)  # shape is still 1D

    # 3) Reshape to 128×128 (or your actual image shape).
    data_float = data_float.reshape((128, 128))

    # 4) Display
    plt.imshow(data_float, cmap='gray')
    plt.title("Y_test.bin Decoded Image (float)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
