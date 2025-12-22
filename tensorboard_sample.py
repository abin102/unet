import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import time

# --- Step 1: Create random data ---
x = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randint(0, 2, (100,))  # binary labels

# --- Step 2: Define a simple model ---
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TinyNet()

# --- Step 3: Define optimizer and loss ---
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# --- Step 4: Create a TensorBoard writer ---
writer = SummaryWriter(log_dir=f"runs/toy_experiment_{int(time.time())}")

# --- Step 5: Log the model graph ---
writer.add_graph(model, x)

# --- Step 6: Training loop with logging ---
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Log scalar value (loss)
    writer.add_scalar("Loss/train", loss.item(), epoch)

    # Log histogram of weights
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    # Log random images for visualization (dummy)
    img_grid = vutils.make_grid(torch.randn(16, 3, 32, 32))
    writer.add_image("Random Images", img_grid, epoch)

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

# --- Step 7: Close the writer ---
writer.close()

# --- Step 6: Training loop with logging ---
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # Log scalar value (loss)
    writer.add_scalar("Loss/train", loss.item(), epoch)

    # Log histogram of weights
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    # Log random images for visualization (dummy)
    img_grid = vutils.make_grid(torch.randn(16, 3, 32, 32))
    writer.add_image("Random Images", img_grid, epoch)

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

# --- Step 7: Close the writer ---
writer.close()
