import os
from pathlib import Path
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from loguru import logger
import json



def get_isic_transforms(image_size=224,
                        imagenet_norm=True,
                        mean=None,
                        std=None,
                        train=True):
    """
    Creates a composed torchvision transform pipeline specifically for ISIC data.
    """

    # 1. Determine Normalization
    if imagenet_norm:
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif mean is not None and std is not None:
        norm = T.Normalize(mean, std)
    else:
        # Fallback, though normalization is strongly recommended
        norm = T.Lambda(lambda x: x) # Identity

    pipeline = []

    # 2. Create the specific pipeline for Train or Val
    if train:
        # --- Training Transforms (with Augmentation) ---
        pipeline.extend([
            # 1. Resize and crop to a fixed size.
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),

            # 2. Geometric augmentations: lesions have no "up" direction.
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90), # Rotate by up to 90 degrees

            # 3. Color augmentations: Simulates different lighting & skin tones.
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

            # 4. Convert to tensor and normalize
            T.ToTensor(),
            norm
        ])
    else:
        # --- Validation/Test Transforms (Preprocessing Only) ---
        
        # 1. Resize the smallest edge to be slightly larger than the crop size.
        # e.g., if image_size=224, resize to 256.
        resize_size = int(image_size * 1.15) 
        if resize_size < image_size: # safety check
             resize_size = image_size

        pipeline.extend([
            T.Resize(resize_size),
            T.CenterCrop(image_size), # Take a deterministic crop from the center
            T.ToTensor(),
            norm
        ])

    return T.Compose(pipeline)


class ManifestImageDataset(Dataset):
    """
    Minimal Dataset that reads a manifest CSV with columns: image_file,label,class_id
    manifest_path: path to manifest CSV
    images_dir: folder containing image files referenced by image_file
    transform: torchvision transforms applied to PIL image
    """
    def __init__(self, manifest_path, images_dir, transform=None, difficulty_map_path=None):
        self.manifest_path = Path(manifest_path)
        self.images_dir = Path(images_dir)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        if not self.images_dir.exists():
            raise FileNotFoundError(self.images_dir)

        self.samples = []
        with open(self.manifest_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['image_file'], int(row['class_id'])))

        if not self.samples:
            raise RuntimeError("Manifest is empty!")

        label_set = sorted({int(s[1]) for s in self.samples})
        logger.info("Unique integer class_ids in manifest: {}", label_set)
        # load mapping JSON once — this isic2018_class_map.json must contain {name: idx}
        mapping_file = self.manifest_path.parent / "isic2018_class_map.json"
        with open(mapping_file, "r") as f:
            name_to_idx = json.load(f)  # {"AKIEC":0, ...}

        # normalize mapping
        name_to_idx = {str(k): int(v) for k, v in name_to_idx.items()}

        # build reverse mapping
        idx_to_name = {int(v): str(k) for k, v in name_to_idx.items()}

        # expose metadata
        self.class_to_idx = name_to_idx         # {"AKIEC":0, ...}
        self.idx_to_name  = idx_to_name         # {0:"AKIEC", ...}
        self.classes      = [idx_to_name[i] for i in sorted(idx_to_name.keys())]  # ordered

        logger.info("Loaded class_to_idx: {}", self.class_to_idx)
        logger.info("Loaded classes: {}", self.classes)


        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])

        # --- NEW: Safe Loading of Difficulty Map ---
        logger.debug(f"DEBUG: Initializing Dataset. Difficulty path received: '{difficulty_map_path}'")
        self.difficulty_map = None
        if difficulty_map_path and Path(difficulty_map_path).exists():
            with open(difficulty_map_path, 'r') as f:
                raw_map = json.load(f)
            
            # Normalize to integers: 0=Hard, 1=Medium, 2=Easy
            mapping = {"hard": 0, "medium": 1, "easy": 2}
            self.difficulty_map = {}
            
            for k, v in raw_map.items():
                # Handle both string keys ("hard") and int keys (0)
                if isinstance(v, str):
                    self.difficulty_map[k] = mapping.get(v.lower(), 1)
                else:
                    self.difficulty_map[k] = int(v)
            logger.info(f"Difficulty map loaded. Tracking {len(self.difficulty_map)} images.")
        else:
            # If path is None or file doesn't exist, we stay in "Legacy Mode"
            self.difficulty_map = None



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, class_id = self.samples[idx]
        p = self.images_dir / img_file
        img = Image.open(p).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        # --- BACKWARD COMPATIBILITY LOGIC ---
        if self.difficulty_map is not None:
            # New behavior: Return 3 items
            diff_id = self.difficulty_map.get(str(img_file), 1) # Default to Medium
            return img, class_id, diff_id
        else:
            # Old behavior: Return 2 items (Old models won't break)
            return img, class_id


