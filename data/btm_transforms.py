# dataloaders/btm_transforms.py

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_btm_transform(
    train: bool,
    image_size: int = 224,
    imagenet_norm: bool = True,
    mean=None,
    std=None,
):
    """
    Build BTM-specific transforms.

    TRAIN:
      - RandomResizedCrop(224, scale=(0.2, 1.0))
      - RandomApply(ColorJitter(...), p=0.8)
      - RandomRotation(45)
      - RandomApply(GaussianBlur, p=0.5)
      - RandomHorizontalFlip()
      - ToTensor()
      - Normalize(mean, std)

    TEST:
      - Resize((224, 224))
      - ToTensor()
      - Normalize(mean, std)
    """
    if imagenet_norm:
        norm_mean, norm_std = IMAGENET_MEAN, IMAGENET_STD
    elif mean is not None and std is not None:
        norm_mean, norm_std = mean, std
    else:
        # fall back to identity (no normalization)
        norm_mean, norm_std = None, None

    if train:
        tf_list = [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomApply(
                [T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )],
                p=0.8,
            ),
            T.RandomRotation(45),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    else:
        tf_list = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]

    if norm_mean is not None and norm_std is not None:
        tf_list.append(T.Normalize(mean=norm_mean, std=norm_std))

    return T.Compose(tf_list)
