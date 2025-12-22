# ensure project root is on sys.path so "from data import REGISTRY" works
# ensure project root is importable so "from data import REGISTRY" works
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # one level up from scripts/
sys.path.insert(0, str(PROJECT_ROOT))
import random
from loguru import logger

from typing import Iterable, Optional, List
import torchvision as tv
from .cifar10v2 import CIFAR10V2
import os
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset


REGISTRY = {}

def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco
def _make_transforms(image_size=224,
                     to_rgb=False,
                     imagenet_norm=True,
                     mean=None,
                     std=None,
                     train=True):
    """
    Build torchvision transforms with optional ImageNet or custom normalization.
    Prints which normalization is applied for transparency.
    """
    t = []
    if to_rgb:
        t.append(tv.transforms.Grayscale(num_output_channels=3))

    t.append(tv.transforms.Resize(image_size))

    if train:
        t.extend([
            tv.transforms.RandomCrop(image_size, padding=4),
            tv.transforms.RandomHorizontalFlip()
        ])

    t.append(tv.transforms.ToTensor())

    # normalization logic with info print
    if imagenet_norm:
        print(f"[INFO] Applying ImageNet normalization (mean={ [0.485, 0.456, 0.406] }, std={ [0.229, 0.224, 0.225] })")
        t.append(tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]))
    elif mean is not None and std is not None:
        print(f"[INFO] Applying custom normalization (mean={mean}, std={std})")
        t.append(tv.transforms.Normalize(mean=mean, std=std))
    else:
        print(f"[INFO] No normalization applied (imagenet_norm=False and no custom mean/std provided).")

    return tv.transforms.Compose(t)



@register("mnist")
def mnist(data_dir, image_size=224, to_rgb=True, imagenet_norm=True, mean=None, std=None):
    """
    MNIST registry entry. Accepts mean/std for compatibility with configs that supply custom normalization.
    Uses train=True for training transforms so augmentations (RandomCrop/Flip) apply.
    """
    tf_train = _make_transforms(image_size=image_size, to_rgb=to_rgb,
                                imagenet_norm=imagenet_norm, mean=mean, std=std, train=True)
    tf_test  = _make_transforms(image_size=image_size, to_rgb=to_rgb,
                                imagenet_norm=imagenet_norm, mean=mean, std=std, train=False)

    train = tv.datasets.MNIST(data_dir, train=True, download=True, transform=tf_train)
    test  = tv.datasets.MNIST(data_dir, train=False, download=True, transform=tf_test)
    return train, test

@register("cifar10")
def cifar10(data_dir, image_size=224, to_rgb=False, imagenet_norm=False,
            mean=None, std=None):
    tf_train = _make_transforms(image_size, to_rgb, imagenet_norm, train=True,
                                mean=mean, std=std)
    tf_test  = _make_transforms(image_size, to_rgb, imagenet_norm, train=False,
                                mean=mean, std=std)
    train = tv.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf_train)
    test  = tv.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_test)
    return train, test



@register("cifar10v2")
def cifar10v2(data_dir, image_size=32, to_rgb=False, imagenet_norm=False,
              mean=None, std=None, batch_size=128, class_balance=False, imb_factor=1.0,
              imb_type='exp'):
    """  
    Normalization is handled inside CIFAR10V2 class.
    
    """
    ds = CIFAR10V2(batch_size=batch_size, class_balance=class_balance, imb_factor=imb_factor)
    return ds.train.dataset, ds.test.dataset


# ... existing REGISTRY and register() definitions ...

@register("firerisk")
def firerisk(data_dir, image_size=224, to_rgb=False, imagenet_norm=True, train=True, **kwargs):
    tf_train = _make_transforms(image_size=image_size, to_rgb=to_rgb,
                                imagenet_norm=imagenet_norm, train=True)
    tf_test = _make_transforms(image_size=image_size, to_rgb=to_rgb,
                               imagenet_norm=imagenet_norm, train=False)

    train = tv.datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=tf_train)
    test = tv.datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=tf_test)
    return train, test



@register("cifar100_imb")
def cifar100_imb(data_dir,
                 image_size=224,
                 to_rgb=False,
                 imagenet_norm=True,
                 mode='none',             # 'none' | 'even' | 'odd' | 'custom' | 'exp'
                 downsample_frac=1.0,     # used by even/odd/custom (fraction to keep)
                 custom_classes: Optional[Iterable[int]] = None,     # list when mode=='custom'
                 seed=42,
                 imb_factor: Optional[float] = None,    # for mode=='exp', e.g. 0.01
                 imbalance_ratio: Optional[float] = None, # alternatively for mode=='exp', e.g. 100
                 train=True):
    """
    Create CIFAR-100 training dataset with controlled imbalance.
    Returns: (train_dataset, test_dataset)
    """
    # transforms
    tf_train = _make_transforms(image_size, to_rgb, imagenet_norm, train=True)
    tf_test  = _make_transforms(image_size, to_rgb, imagenet_norm, train=False)

    # load full CIFAR100 splits
    full_train = CIFAR100(root=data_dir, train=True, download=True, transform=tf_train)
    test = CIFAR100(root=data_dir, train=False, download=True, transform=tf_test)

    # quick exit only if explicitly 'none' (do NOT short-circuit on downsample_frac here)
    if mode == 'none':
        return full_train, test

    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Mode: exponential long-tailed
    # ------------------------------------------------------------------
    if mode == 'exp':
        # CIFAR-100 train has 500 images / class
        cls_num = 100
        img_max = 500  # original number of images per class in CIFAR-100

        if imbalance_ratio is not None:
            if imbalance_ratio <= 1:
                raise ValueError("imbalance_ratio must be > 1")
            imb_factor_use = 1.0 / float(imbalance_ratio)
        else:
            # default if user didn't provide anything
            if imb_factor is None:
                imb_factor_use = 0.01   # common default => smallest class ~5 images when img_max=500
            else:
                if not (0.0 < imb_factor <= 1.0):
                    raise ValueError("imb_factor must be in (0, 1]. To specify e.g. 100x imbalance use imbalance_ratio=100")
                imb_factor_use = float(imb_factor)

        # compute per-class image counts using exponential rule:
        # n_i = img_max * (imb_factor_use ** (i/(cls_num-1)))  for i=0..cls_num-1
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            n_i = img_max * (imb_factor_use ** (cls_idx / float(cls_num - 1)))
            img_num_per_cls.append(max(1, int(round(n_i))))  # ensure at least 1

        # build map class -> list of indices
        class_to_indices = {c: [] for c in range(cls_num)}
        for idx, (_, label) in enumerate(full_train):
            class_to_indices[label].append(idx)

        keep_indices: List[int] = []
        for c in range(cls_num):
            orig_idxs = class_to_indices[c]
            keep_n = min(len(orig_idxs), img_num_per_cls[c])
            chosen = rng.sample(orig_idxs, keep_n)
            keep_indices.extend(chosen)

        keep_indices.sort()
        train_subset = Subset(full_train, keep_indices)

        # # DEBUG: print created counts for verification
        # from collections import Counter
        # counts = Counter()
        # for idx in keep_indices:
        #     _, lbl = full_train[idx]
        #     counts[int(lbl)] += 1
        # print("DEBUG cifar100_imb (exp) created counts:", dict(sorted(counts.items())))

        return train_subset, test

    # ------------------------------------------------------------------
    # Modes: even, odd, custom (existing behavior; uses downsample_frac)
    # ------------------------------------------------------------------
    if mode not in ('even', 'odd', 'custom'):
        raise ValueError(f"Unsupported mode: {mode}")

    # determine which classes to downsample
    if mode == 'even':
        target_classes = [c for c in range(100) if (c % 2 == 0)]
    elif mode == 'odd':
        target_classes = [c for c in range(100) if (c % 2 == 1)]
    else:  # custom
        if not custom_classes:
            raise ValueError("custom_classes must be provided when mode='custom'")
        target_classes = list(custom_classes)

    # build map class -> list of indices
    class_to_indices = {c: [] for c in range(100)}
    for idx, (_, label) in enumerate(full_train):
        class_to_indices[label].append(idx)

    keep_indices: List[int] = []
    for c in range(100):
        idxs = class_to_indices[c]
        if c in target_classes:
            # compute how many to keep (orig_n == 500 for CIFAR100 train)
            orig_n = len(idxs)
            keep_n = max(1, int(round(orig_n * float(downsample_frac))))
            chosen = rng.sample(idxs, keep_n)
            keep_indices.extend(chosen)
        else:
            keep_indices.extend(idxs)

    # optional: sort to keep deterministic order
    keep_indices.sort()

    # # DEBUG: print created counts for verification
    # from collections import Counter
    # counts = Counter()
    # for idx in keep_indices:
    #     _, lbl = full_train[idx]
    #     counts[int(lbl)] += 1
    # print("DEBUG cifar100_imb (even/odd/custom) created counts:", dict(sorted(counts.items())))

    train_subset = Subset(full_train, keep_indices)
    return train_subset, test



@register("tiny_imagenet")
def tiny_imagenet(data_dir, image_size=64, to_rgb=False, imagenet_norm=True, train=True, **kwargs):
    """
    Tiny-ImageNet registry. Keeps default image_size=64 but you can set image_size=224 to use pretrained backbones.
    Expects folder:
      data_dir/train/<wnid>/*
      data_dir/val/<wnid>/*
    """
    tf_train = _make_transforms(image_size=image_size, to_rgb=to_rgb,
                                imagenet_norm=imagenet_norm, train=True)
    tf_test  = _make_transforms(image_size=image_size, to_rgb=to_rgb,
                                imagenet_norm=imagenet_norm, train=False)

    train = tv.datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=tf_train)
    val   = tv.datasets.ImageFolder(root=os.path.join(data_dir, "val"),   transform=tf_test)
    return train, val



@register("isic2018")
def isic2018(data_dir,
             image_size=224,
             imagenet_norm=True,
             to_rgb=False,
             train=True,
             mean=None,
             std=None,
             **kwargs):
    """
    ISIC2018 registry entry using prebuilt manifest CSVs.
    Accepts mean/std for compatibility with other datasets.
    """
    import os
    import torchvision as tv
    from .isic2018_data_manifest import ManifestImageDataset, get_isic_transforms  # ensure this import exists

    # --- 1. EXTRACT THE MAP PATH ---
    # If the key is missing (old configs), this becomes None.
    diff_map_path = kwargs.get("difficulty_map_path", None)
    logger.debug("ISIC2018 dataset registry: difficulty_map_path={}", diff_map_path)
    # transforms for train/test
    tf_train = get_isic_transforms(image_size=image_size,
                                imagenet_norm=imagenet_norm,
                                mean=mean, std=std,
                                train=True)
    tf_test = get_isic_transforms(image_size=image_size,
                               imagenet_norm=imagenet_norm,
                               mean=mean, std=std,
                               train=False)

    # manifest paths
    train_manifest = os.path.join(data_dir, "isic2018_train_manifest.csv")
    val_manifest   = os.path.join(data_dir, "isic2018_val_manifest.csv")
    test_manifest  = os.path.join(data_dir, "isic2018_test_manifest.csv")

    # image folders
    train_img_dir = os.path.join(data_dir, "training", "images")
    val_img_dir   = os.path.join(data_dir, "validation", "images")
    test_img_dir  = os.path.join(data_dir, "test", "images")

    # instantiate datasets
    train_ds = ManifestImageDataset(train_manifest, train_img_dir, transform=tf_train)
    # val_ds   = ManifestImageDataset(val_manifest,   val_img_dir,   transform=tf_test)
    test_ds  = ManifestImageDataset(test_manifest,  test_img_dir,  transform=tf_test,
                                    difficulty_map_path=diff_map_path)

    return train_ds, test_ds




@register("btm")
def btm(
    data_dir,
    image_size=224,
    imagenet_norm=True,
    to_rgb=False,   # kept for API consistency; ignored here
    train=True,     # not actually used; function returns train & test
    mean=None,
    std=None,
    **kwargs,
):
    """
    BTM dataset registry.

    Expects:
      data_dir/training/<class>/*.png|jpg
      data_dir/testing/<class>/*.png|jpg
    """
    from .btm_transforms import make_btm_transform

    # build transforms
    tf_train = make_btm_transform(
        train=True,
        image_size=image_size,
        imagenet_norm=imagenet_norm,
        mean=mean,
        std=std,
    )
    tf_test = make_btm_transform(
        train=False,
        image_size=image_size,
        imagenet_norm=imagenet_norm,
        mean=mean,
        std=std,
    )

    train_root = os.path.join(data_dir, "training")
    test_root  = os.path.join(data_dir, "testing")

    train_ds = tv.datasets.ImageFolder(root=train_root, transform=tf_train)
    test_ds  = tv.datasets.ImageFolder(root=test_root,  transform=tf_test)

    return train_ds, test_ds
