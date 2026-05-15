"""
Dataset sanity checks:
  1. Random 20-sample grid: raw image / mask / overlay
  2. Mask value range (should be only 0 and 1)
  3. Image size consistency across splits
  4. Transform alignment (image vs mask after augmentation)
"""

import random
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import CrackDataset, get_transforms

ROOT = "CRACK500"
SEED = 42
DIAGNOSTICS_DIR = Path("generated/diagnostics")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 1. Raw visualization: 20 random samples ──────────────────────────────────
def plot_raw_samples(ds, n=20, out=DIAGNOSTICS_DIR / "check_raw_samples.png"):
    indices = random.sample(range(len(ds)), n)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 2))
    fig.suptitle("Raw samples — image / mask / overlay", fontsize=13)

    for row, idx in enumerate(indices):
        img, mask = ds.get_raw(idx)
        overlay = img.copy()
        overlay[mask == 1] = (overlay[mask == 1] * 0.4 + np.array([255, 0, 0]) * 0.6).clip(0, 255).astype(np.uint8)

        axes[row, 0].imshow(img)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Image")

        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title("Mask")

        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")
        if row == 0:
            axes[row, 2].set_title("Overlay")

    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=80, bbox_inches="tight")
    print(f"[1] Raw sample grid saved → {out}")


# ── 2. Mask value check ───────────────────────────────────────────────────────
def check_mask_values(ds, n=200):
    bad = []
    all_vals = set()
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    for idx in indices:
        _, mask = ds.get_raw(idx)
        vals = set(np.unique(mask).tolist())
        all_vals |= vals
        if not vals.issubset({0, 1}):
            bad.append((idx, vals))

    status = "OK" if not bad else f"FAIL — {len(bad)} bad masks"
    print(f"[2] Mask value check ({n} samples): {status}")
    print(f"    All unique values seen: {sorted(all_vals)}")
    if bad:
        for idx, vals in bad[:5]:
            print(f"    idx={idx}  values={vals}")


# ── 3. Image size consistency ─────────────────────────────────────────────────
def check_sizes(ds, n=300):
    sizes = {}
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    for idx in indices:
        img, mask = ds.get_raw(idx)
        h, w = img.shape[:2]
        mh, mw = mask.shape[:2]
        if img.shape[:2] != mask.shape[:2]:
            print(f"    [MISMATCH] idx={idx}  img={img.shape[:2]}  mask={mask.shape[:2]}")
        key = (h, w)
        sizes[key] = sizes.get(key, 0) + 1

    print(f"[3] Size distribution across {n} samples:")
    for size, cnt in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"    {size[0]}x{size[1]} — {cnt} samples")


# ── 4. Transform alignment check ─────────────────────────────────────────────
def check_transform_alignment(
    ds_train,
    n=20,
    out=DIAGNOSTICS_DIR / "check_transform_alignment.png",
):
    indices = random.sample(range(len(ds_train)), n)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 2))
    fig.suptitle("After train transforms — image / mask / overlay", fontsize=13)

    for row, idx in enumerate(indices):
        img_t, mask_t = ds_train[idx]
        # denormalize
        img_np = img_t.permute(1, 2, 0).numpy() * std + mean
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        mask_np = mask_t.numpy().astype(np.uint8)

        overlay = img_np.copy()
        overlay[mask_np == 1] = (overlay[mask_np == 1] * 0.4 + np.array([255, 0, 0]) * 0.6).clip(0, 255).astype(np.uint8)

        axes[row, 0].imshow(img_np);  axes[row, 0].axis("off")
        axes[row, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1); axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay); axes[row, 2].axis("off")
        if row == 0:
            axes[row, 0].set_title("Image (transformed)")
            axes[row, 1].set_title("Mask (transformed)")
            axes[row, 2].set_title("Overlay")

        # verify mask tensor values
        unique = mask_t.unique().tolist()
        if not set(unique).issubset({0, 1}):
            print(f"    [WARN] idx={idx} post-transform mask has values: {unique}")

    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=80, bbox_inches="tight")
    print(f"[4] Transform alignment grid saved → {out}")


# ── 5. Quick class balance ────────────────────────────────────────────────────
def check_class_balance(ds, n=500):
    crack_px = total_px = 0
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    for idx in indices:
        _, mask = ds.get_raw(idx)
        crack_px += int(mask.sum())
        total_px += mask.size
    ratio = crack_px / total_px * 100
    print(f"[5] Class balance ({n} samples): crack={ratio:.2f}%  background={100-ratio:.2f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Loading datasets …")
    ds_train_raw = CrackDataset(ROOT, split="train", transform=None)
    ds_train_aug = CrackDataset(ROOT, split="train")
    ds_val       = CrackDataset(ROOT, split="val",   transform=None)
    ds_test      = CrackDataset(ROOT, split="test",  transform=None)
    print(f"  train={len(ds_train_raw)}  val={len(ds_val)}  test={len(ds_test)}")
    print("=" * 60)

    plot_raw_samples(ds_train_raw, n=20)
    check_mask_values(ds_train_raw, n=200)
    check_sizes(ds_train_raw, n=300)
    check_transform_alignment(ds_train_aug, n=20)
    check_class_balance(ds_train_raw, n=500)

    print("=" * 60)
    print("All checks done.")
