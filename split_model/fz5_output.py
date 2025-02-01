import numpy as np
import matplotlib.pyplot as plt

# Load the raw output file
output_file_path = "./fz5/dma_test.bin"
output_data = np.fromfile(output_file_path, dtype=np.uint8)  # Default to uint8

# Debug: Inspect the shape and content
print(f"Output data shape: {output_data.shape}")
print(f"First 10 output data values: {output_data[:10]}")

# The output file may contain padding or unexpected size
# Check if the total size is divisible by 128x128
expected_size = 128 * 128
if output_data.size < expected_size:
    print(f"Warning: Output data is smaller than expected. Expected {expected_size}, got {output_data.size}.")
    padded_output_data = np.zeros(expected_size, dtype=np.uint8)
    padded_output_data[:output_data.size] = output_data
    output_data = padded_output_data
elif output_data.size > expected_size:
    print(f"Warning: Output data is larger than expected. Expected {expected_size}, got {output_data.size}.")
    output_data = output_data[:expected_size]

# Reshape to 128x128
try:
    output_image = output_data.reshape((128, 128))
except ValueError as e:
    print(f"Error reshaping data: {e}")
    print("Ensure the output data matches the expected 128x128 size.")
    exit()

# Display the image
plt.imshow(output_image, cmap="gray")
plt.title("Output Image")
plt.axis("off")
plt.show()
