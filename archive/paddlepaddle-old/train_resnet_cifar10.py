# File: archive/paddlepaddle-old/train_resnet_cifar10.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision.models import resnet50
from paddle.metric import Accuracy

# Define transformations for the dataset
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
train_dataset = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.Cifar10(mode='test', transform=transform)

# Define the model, loss function, and optimizer
model = resnet50(num_classes=10)  # Adjust for CIFAR-10 (10 classes)
criterion = nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# Set up the data loader
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the training loop

# ---------------------------------------------------
# CHANGE BATCHES AND EPOCHS HERE FOR ACTUALL LEARNING
max_batches = 1
epochs = 1
# ---------------------------------------------------

for epoch in range(epochs):
    model.train()
    for batch_id, (images, labels) in enumerate(train_loader):
        if batch_id >= max_batches:
            break  # Stop after processing max_batches batches

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if batch_id % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_id}], Loss: {loss.numpy()}")

    # Evaluate the model on test data
    model.eval()
    acc = Accuracy()
    for images, labels in test_loader:
        outputs = model(images)
        acc.update(outputs, labels)
    print(f"Epoch [{epoch+1}/{epochs}], Test Accuracy: {acc.accumulate()}")
    acc.reset()

print("Training complete.")

# Assuming the model has been trained as shown previously
# Save with paddle.jit.save
paddle.jit.save(
    layer=model,
    path='resnet50_cifar10/model',
    input_spec=[paddle.static.InputSpec(shape=[None, 3, 32, 32], dtype='float32')]
)
