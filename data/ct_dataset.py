import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Dataset
from data import REGISTRY  

class CT_NpyDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=512, **kwargs):
        self.split_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.split_dir, 'images')
        self.masks_dir = os.path.join(self.split_dir, 'masks')
        self.files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.npy')])
        self.image_size = image_size
        
        # Only augment if we are in the 'train' split
        self.augment = (split == 'train')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        image = np.load(os.path.join(self.images_dir, file_name))
        mask = np.load(os.path.join(self.masks_dir, file_name))
        
        # 1. Preprocessing: Windowing & Norm
        # Lung Window: -1000 to 500
        image = np.clip(image, -1000, 500)
        image = (image - (-1000)) / (500 - (-1000))
        
        # 2. Convert to Tensor
        image = torch.from_numpy(image).float().unsqueeze(0) # (1, H, W)
        mask = torch.from_numpy(mask).long()                 # (H, W) -> will squeeze later if needed

        # 3. SAFETY RESIZE (Crucial for mixed datasets)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size), 
                                  mode='bilinear', align_corners=False).squeeze(0)
            
            # Masks MUST use 'nearest' to keep class integers (0,1,2)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(self.image_size, self.image_size), 
                                 mode='nearest').long().squeeze(0).squeeze(0)

        # 4. SYNCHRONIZED AUGMENTATION (Train only)
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random Rotation (-15 to +15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                # Mask must use NEAREST to preserve integers
                mask = TF.rotate(mask.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        return image, mask

def build_ct_dataset(data_dir, **kwargs):
    # Pass the split explicitly so augmentations only happen on train
    return CT_NpyDataset(data_dir, split='train', **kwargs), \
           CT_NpyDataset(data_dir, split='val', **kwargs)

REGISTRY["ct_segmentation"] = build_ct_dataset