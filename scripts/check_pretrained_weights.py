import torch
import torchvision as tv
import torch.nn.functional as F
import numpy as np
import sys, os

# make sure Python sees your model registry
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import REGISTRY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n🔍 Testing CIFAR-stem interpolation in resnet50_torch...\n")

# --- 1. Load torchvision official pretrained ResNet50 weights ---
try:
    src_weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
    src_model = tv.models.resnet50(weights=src_weights)
    src_state = src_model.state_dict()
except Exception:
    src_model = tv.models.resnet50(pretrained=True)
    src_state = src_model.state_dict()

src_conv1 = src_state["conv1.weight"].clone()
print(f"[torchvision] conv1.shape = {tuple(src_conv1.shape)}, mean={src_conv1.mean():.4f}, std={src_conv1.std():.4f}")

# --- 2. Instantiate your registry model with cifar_stem=True, pretrained=True ---
model_cifar = REGISTRY["resnet50_torch"](num_classes=10, pretrained=True, cifar_stem=True)
model_cifar = model_cifar.to(device)

own_conv1 = model_cifar.conv1.weight.detach().cpu()
print(f"[registry-cifar] conv1.shape = {tuple(own_conv1.shape)}, mean={own_conv1.mean():.4f}, std={own_conv1.std():.4f}")

# --- 3. Check correlation between interpolated conv1 and original ImageNet conv1 ---
# Resize ImageNet conv1 to 3x3 for fair comparison (same as your code)
w_src_r = src_conv1.view(src_conv1.size(0)*src_conv1.size(1), 1, 7, 7)
w_src_resized = F.interpolate(w_src_r, size=(3,3), mode='bilinear', align_corners=False)
w_src_resized = w_src_resized.view(src_conv1.size(0), src_conv1.size(1), 3, 3)

# Compute cosine similarity between flattened tensors
def cos_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

similarity = cos_sim(own_conv1, w_src_resized)
print(f"✅ Cosine similarity between interpolated (expected) and model.conv1 = {similarity:.4f}")

# --- 4. Also instantiate non-cifar-stem model to confirm conv1 is 7x7 and copied directly ---
model_normal = REGISTRY["resnet50_torch"](num_classes=10, pretrained=True, cifar_stem=False)
own_conv1_normal = model_normal.conv1.weight.detach().cpu()
print(f"[registry-normal] conv1.shape = {tuple(own_conv1_normal.shape)}, mean={own_conv1_normal.mean():.4f}, std={own_conv1_normal.std():.4f}")

# --- 5. Sanity check summary ---
print("\n--- SUMMARY ---")
print(f"conv1 copied/interpolated OK? {'YES' if similarity > 0.85 else 'Probably NO'}")
print(f"cifar_stem conv1 kernel: {tuple(own_conv1.shape)} | normal stem conv1 kernel: {tuple(own_conv1_normal.shape)}")
print(f"conv1 weight mean/std diff (cifar vs normal): {abs(own_conv1.mean()-own_conv1_normal.mean()):.4f} / {abs(own_conv1.std()-own_conv1_normal.std()):.4f}")

# --- Optional: quick forward sanity check on random CIFAR image ---
x = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    y = model_cifar(x)
print(f"Forward OK, output shape: {tuple(y.shape)}")
