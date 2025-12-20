# train.py
)
# File: train.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

# train.py
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from dataloader import SyntheticDataset  # Import the dataset
from model import UNet, visualize_sample, IMG_HEIGHT, IMG_WIDTH  # Import IMG_HEIGHT and IMG_WIDTH

# Instantiate dataset and dataloader
train_dataset = SyntheticDataset(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define model, loss, and optimizer
model = UNet(num_classes=1)  # Update to single output channel
criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# Retrieve the first sample from the dataset
first_image, first_mask = train_dataset[0]

# Visualize
visualize_sample(first_image, first_mask)

# ---------------------------------------------------
# CHANGE BATCHES AND EPOCHS HERE FOR ACTUALL LEARNING
max_batches = 1
epochs = 1
# ---------------------------------------------------

# Training loop
for epoch in range(epochs):
    model.train()
    for batch_id, (images, masks) in enumerate(train_loader):
        if batch_id >= max_batches:
            break  # Stop after processing max_batches batches
        print(f"Batch {batch_id} - images shape: {images.shape}")

        # Forward pass
        outputs = model(images)
        print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")

        # Reshape masks to [batch_size, 1, height, width] for BCEWithLogitsLoss
        masks = paddle.unsqueeze(masks, axis=1).astype('float32')

        # Calculate loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if batch_id % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_id}], Loss: {loss.numpy()}")

print("Training complete.")

# Export the trained model
paddle.jit.save(
    layer=model,
    path='unet_model/unet',
    input_spec=[paddle.static.InputSpec(shape=[None, 3, IMG_HEIGHT, IMG_WIDTH], dtype='float32')]
)
