# import sys
# sys.path.append("..")  # add project root
import torch
from models import REGISTRY as M
from data import REGISTRY as D

cfg = {
    "data_dir": "data/tiny-imagenet",
    "image_size": 64,
    "to_rgb": False,
    "imagenet_norm": True
}
# build dataloader
train_ds, val_ds = D["tiny_imagenet"](cfg["data_dir"], image_size=cfg["image_size"], to_rgb=cfg["to_rgb"], imagenet_norm=cfg["imagenet_norm"])
loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
images, targets = next(iter(loader))
# build model (pretrained copy only matching keys)
model = M["resnet50_torch"](num_classes=200, pretrained=True, cifar_stem=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images = images.to(device)
with torch.no_grad():
    out = model(images)
print("out.shape:", out.shape)  # -> (8,200)
