import torch
import torch.optim as optim
import torch.nn as nn

# Suppose you have a PyTorch dataset that returns (image, mask)
# from a custom dataset class or transforms
train_dataset = ...  # your dataset in PyTorch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

model = UNetBrevitas(in_channels=3, out_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(5):
    for batch_id, (images, masks) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)  # [N, 1, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        if batch_id % 10 == 0:
            print(f"Epoch [{epoch+1}/5], Batch [{batch_id}], Loss: {loss.item():.4f}")
