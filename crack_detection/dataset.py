import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_photometric_transforms(augmentation_profile: str = "baseline"):
    if augmentation_profile == "baseline":
        return [
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        ]

    if augmentation_profile == "mild":
        return [
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.35,
                        contrast=0.35,
                        saturation=0.25,
                        hue=0.06,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=18,
                        val_shift_limit=15,
                        p=1.0,
                    ),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        ensure_safe_range=True,
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(85, 120), p=1.0),
                    A.CLAHE(clip_limit=(1.0, 2.5), p=1.0),
                ],
                p=0.45,
            ),
        ]

    if augmentation_profile == "strong":
        return [
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.3,
                        hue=0.08,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=12,
                        sat_shift_limit=25,
                        val_shift_limit=20,
                        p=1.0,
                    ),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.35,
                        contrast_limit=0.35,
                        ensure_safe_range=True,
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(70, 140), p=1.0),
                    A.CLAHE(clip_limit=(1.0, 4.0), p=1.0),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.RandomShadow(
                        shadow_roi=(0.0, 0.0, 1.0, 1.0),
                        num_shadows_limit=(1, 3),
                        shadow_dimension=5,
                        shadow_intensity_range=(0.25, 0.6),
                        p=1.0,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.4, -0.1),
                        contrast_limit=(-0.1, 0.15),
                        ensure_safe_range=True,
                        p=1.0,
                    ),
                ],
                p=0.35,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.25,
            ),
        ]

    raise ValueError(
        f"Unsupported augmentation profile: {augmentation_profile}. "
        "Choose from 'baseline', 'mild', or 'strong'."
    )


def get_transforms(
    split: str,
    img_size: int = 360,
    augmentation_profile: str = "baseline",
    use_random_resized_crop: bool = True,
):
    size = (img_size, img_size)
    if split == "train":
        transforms = []
        if use_random_resized_crop:
            transforms.append(A.RandomResizedCrop(size=size, scale=(0.5, 1.0)))
        else:
            transforms.append(A.Resize(height=img_size, width=img_size))

        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                *build_photometric_transforms(augmentation_profile),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
        return A.Compose(transforms)

    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def foreground_aware_square_crop(
    image: np.ndarray,
    mask: np.ndarray,
    min_scale: float = 0.55,
    max_scale: float = 1.0,
):
    if not (0 < min_scale <= max_scale <= 1.0):
        raise ValueError("Foreground crop scale must satisfy 0 < min_scale <= max_scale <= 1.0.")

    height, width = mask.shape
    crop_size = int(round(min(height, width) * random.uniform(min_scale, max_scale)))
    crop_size = max(32, min(crop_size, min(height, width)))

    fg_y, fg_x = np.where(mask > 0)
    if len(fg_y) > 0:
        chosen = random.randrange(len(fg_y))
        center_y = int(fg_y[chosen])
        center_x = int(fg_x[chosen])
        jitter = max(8, crop_size // 6)
        center_y += random.randint(-jitter, jitter)
        center_x += random.randint(-jitter, jitter)
    else:
        center_y = random.randint(0, height - 1)
        center_x = random.randint(0, width - 1)

    top = int(np.clip(center_y - crop_size // 2, 0, height - crop_size))
    left = int(np.clip(center_x - crop_size // 2, 0, width - crop_size))

    return (
        image[top: top + crop_size, left: left + crop_size],
        mask[top: top + crop_size, left: left + crop_size],
    )


class CrackDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        img_size: int = 360,
        augmentation_profile: str = "baseline",
        foreground_crop_prob: float = 0.0,
        foreground_crop_min_scale: float = 0.55,
        foreground_crop_max_scale: float = 1.0,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augmentation_profile = augmentation_profile
        self.foreground_crop_prob = foreground_crop_prob if split == "train" else 0.0
        self.foreground_crop_min_scale = foreground_crop_min_scale
        self.foreground_crop_max_scale = foreground_crop_max_scale
        self.transform = transform if transform is not None else get_transforms(
            split,
            img_size,
            augmentation_profile=augmentation_profile,
            use_random_resized_crop=self.foreground_crop_prob <= 0,
        )

        if not (0.0 <= self.foreground_crop_prob <= 1.0):
            raise ValueError("foreground_crop_prob must be between 0.0 and 1.0.")

        txt = self.root / f"{split}.txt"
        self.samples = []
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.samples.append((self.root / parts[0], self.root / parts[1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        # PIL mode '1' (bool) → uint8 0/1
        mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)

        if self.foreground_crop_prob > 0 and random.random() < self.foreground_crop_prob:
            image, mask = foreground_aware_square_crop(
                image,
                mask,
                min_scale=self.foreground_crop_min_scale,
                max_scale=self.foreground_crop_max_scale,
            )

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image = out["image"]          # (3, H, W) float32 tensor
            mask = out["mask"].long()     # (H, W) int64 tensor

        return image, mask

    def get_raw(self, idx):
        """Return (image_np, mask_np) without any transform, for visualization."""
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)
        return image, mask
