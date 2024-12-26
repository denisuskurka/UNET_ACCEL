import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

from model import UNetBrevitas
from dataset import MyDataset  # the custom dataset defined above

# Desired training image size
HEIGHT = 128
WIDTH = 128

def train_unet_brevitas(
    model,
    data_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs=5,
    log_interval=10
):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [N, 1, 128, 128]
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{batch_idx+1}/{len(data_loader)}], "
                      f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Create real dataset
    root_dir = "./data"  # adjust if needed
    dataset = MyDataset(root_dir, height=HEIGHT, width=WIDTH)
    # If you want a train/val split, you'd do it here
    # e.g. train_size = int(0.8 * len(dataset)), val_size = len(dataset) - train_size
    # train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    # For simplicity, we'll just create one loader for all data
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Create model
    model = UNetBrevitas(in_channels=1, out_channels=1).to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_unet_brevitas(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        device=device,
        num_epochs=5,
        log_interval=10
    )

    print("Training complete!")

    # ------------------------------------------------------------------------
    # Export to QONNX
    # ------------------------------------------------------------------------
    onnx_export_path = "unet_brevitas.onnx"
    model.eval()

    # We need a dummy input for correct shape
    dummy_input = torch.randn(1, 1, HEIGHT, WIDTH).to(device)

    # Export to QONNX
    export_qonnx(
        model, export_path=onnx_export_path, input_t=dummy_input
    )
    # Clean up QONNX
    qonnx_cleanup(onnx_export_path, out_file=onnx_export_path)

    # Convert QONNX to FINN
    finn_model = ModelWrapper(onnx_export_path)
    # If you truly want 'BIPOLAR' input, you can do that here, 
    # or consider e.g. DataType.INT8, etc. (depends on your design)
    finn_model.set_tensor_datatype(finn_model.graph.input[0].name, DataType["BIPOLAR"])
    finn_model = finn_model.transform(ConvertQONNXtoFINN())
    finn_model.save(onnx_export_path)

    print(f"Model saved to {onnx_export_path}")
