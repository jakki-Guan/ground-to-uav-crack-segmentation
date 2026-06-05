import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class SplitSummary:
    domain: str
    split: str
    samples: int
    images_dir: str
    masks_dir: Optional[str]
    split_file: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the repo's paired split-file crack datasets into the "
            "MMSeg/DAFormer CustomDataset layout."
        )
    )
    parser.add_argument("--source-root", default="CRACK500")
    parser.add_argument("--target-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--source-train-split", default="train")
    parser.add_argument("--target-train-split", default="train")
    parser.add_argument("--target-val-split", default="val")
    parser.add_argument("--target-test-split", default="test")
    parser.add_argument("--output-root", default="generated/daformer/crack500_to_uav")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output root.")
    return parser.parse_args()


def read_pair_split(dataset_root: Path, split: str) -> List[Tuple[Path, Path]]:
    split_path = dataset_root / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    pairs: List[Tuple[Path, Path]] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Expected image/mask pair at {split_path}:{line_number}: {line}")
            image_path = dataset_root / parts[0]
            mask_path = dataset_root / parts[1]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            pairs.append((image_path, mask_path))

    if not pairs:
        raise ValueError(f"No samples found in split file: {split_path}")
    return pairs


def save_rgb_png(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").save(dst)


def save_binary_mask(src: Path, dst: Path) -> Dict[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        mask = np.array(image.convert("L"))
    binary = (mask > 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(dst)
    values, counts = np.unique(binary, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def convert_split(
    *,
    domain: str,
    split: str,
    pairs: List[Tuple[Path, Path]],
    domain_root: Path,
    convert_masks: bool,
    write_class_index: bool = False,
) -> SplitSummary:
    image_dir = domain_root / "img_dir" / split
    mask_dir = domain_root / "ann_dir" / split if convert_masks else None
    split_dir = domain_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / f"{split}.txt"
    sample_class_stats = []
    samples_with_class: Dict[int, List[List[Union[str, int]]]] = {0: [], 1: []}

    names = []
    for idx, (image_path, mask_path) in enumerate(pairs):
        name = f"{domain}_{split}_{idx:06d}"
        save_rgb_png(image_path, image_dir / f"{name}.png")
        if mask_dir is not None:
            rel_mask_path = f"ann_dir/{split}/{name}.png"
            counts = save_binary_mask(mask_path, mask_dir / f"{name}.png")
            stats = {"file": rel_mask_path}
            for class_id in (0, 1):
                count = int(counts.get(class_id, 0))
                stats[str(class_id)] = count
                if count > 0:
                    samples_with_class[class_id].append([rel_mask_path, count])
            sample_class_stats.append(stats)
        names.append(name)

    split_file.write_text("\n".join(names) + "\n", encoding="utf-8")
    if write_class_index:
        (domain_root / "sample_class_stats.json").write_text(
            json.dumps(sample_class_stats, indent=2),
            encoding="utf-8",
        )
        (domain_root / "samples_with_class.json").write_text(
            json.dumps({str(key): value for key, value in samples_with_class.items()}, indent=2),
            encoding="utf-8",
        )
    return SplitSummary(
        domain=domain,
        split=split,
        samples=len(names),
        images_dir=str(image_dir),
        masks_dir=str(mask_dir) if mask_dir is not None else None,
        split_file=str(split_file),
    )


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    output_root = Path(args.output_root)

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output root already exists: {output_root}. "
                "Pass --force to rebuild it."
            )
        shutil.rmtree(output_root)

    source_pairs = read_pair_split(source_root, args.source_train_split)
    target_train_pairs = read_pair_split(target_root, args.target_train_split)
    target_val_pairs = read_pair_split(target_root, args.target_val_split)
    target_test_pairs = read_pair_split(target_root, args.target_test_split)

    summaries = [
        convert_split(
            domain="source",
            split="train",
            pairs=source_pairs,
            domain_root=output_root / "source",
            convert_masks=True,
            write_class_index=True,
        ),
        convert_split(
            domain="target",
            split="train",
            pairs=target_train_pairs,
            domain_root=output_root / "target",
            convert_masks=False,
        ),
        convert_split(
            domain="target",
            split="val",
            pairs=target_val_pairs,
            domain_root=output_root / "target",
            convert_masks=True,
        ),
        convert_split(
            domain="target",
            split="test",
            pairs=target_test_pairs,
            domain_root=output_root / "target",
            convert_masks=True,
        ),
    ]

    manifest = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "output_root": str(output_root),
        "format": {
            "img_suffix": ".png",
            "seg_map_suffix": ".png",
            "classes": ["background", "crack"],
            "label_values": {"background": 0, "crack": 1, "ignore": 255},
        },
        "note": (
            "Target train masks are intentionally not converted or referenced by "
            "the DAFormer training config; target labels are used only for val/test reporting."
        ),
        "splits": [asdict(summary) for summary in summaries],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote DAFormer/MMSeg dataset to {output_root}")
    for summary in summaries:
        mask_note = "with masks" if summary.masks_dir is not None else "image-only"
        print(f"{summary.domain}/{summary.split}: {summary.samples} samples ({mask_note})")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
