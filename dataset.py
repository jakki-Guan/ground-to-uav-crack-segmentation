import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split: str, img_size: int = 360):
    size = (img_size, img_size)
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class CrackDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None, img_size: int = 360):
        self.root = Path(root)
        self.split = split
        self.transform = transform if transform is not None else get_transforms(split, img_size)

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
