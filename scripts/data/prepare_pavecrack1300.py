import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_NAMES = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}
MASK_SUFFIX_PATTERN = re.compile(
    r"(?i)(?:[_-](mask|masks|label|labels|gt|ann|annotation|seg|binary))+$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Import the public PaveCrack1300 release into the repository's standard "
            "dataset-root format: images/, masks/, train.txt, val.txt, test.txt, "
            "crossdomain_all.txt, and split_manifest.json."
        )
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Root directory of the extracted raw PaveCrack1300 release.",
    )
    parser.add_argument(
        "--dataset-root",
        default="PaveCrack1300",
        help="Output dataset root in repository-native split-file format.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help=(
            "Optional image directory inside --raw-root. When omitted, the script "
            "tries to auto-discover a likely image folder."
        ),
    )
    parser.add_argument(
        "--masks-dir",
        default=None,
        help=(
            "Optional mask directory inside --raw-root. When omitted, the script "
            "tries to auto-discover a likely mask folder."
        ),
    )
    parser.add_argument("--full-split-name", default="crossdomain_all")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing prepared dataset root.",
    )
    return parser.parse_args()


def resolve_input_dir(
    raw_root: Path,
    provided_value: str | None,
    preferred_names: tuple[str, ...],
    kind: str,
) -> Path:
    if provided_value:
        candidate = Path(provided_value)
        if not candidate.is_absolute():
            candidate = raw_root / candidate
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"{kind} directory not found: {candidate}")
        return candidate

    direct_matches = [
        path for path in raw_root.iterdir()
        if path.is_dir() and path.name.lower() in preferred_names
    ]
    if len(direct_matches) == 1:
        return direct_matches[0].resolve()
    if len(direct_matches) > 1:
        raise RuntimeError(
            f"Ambiguous {kind} directory under {raw_root}: "
            f"{[path.name for path in direct_matches]}. "
            f"Pass --{kind.replace(' ', '-')} explicitly."
        )

    recursive_matches = [
        path for path in raw_root.rglob("*")
        if path.is_dir() and path.name.lower() in preferred_names
    ]
    if len(recursive_matches) == 1:
        return recursive_matches[0].resolve()
    if len(recursive_matches) > 1:
        raise RuntimeError(
            f"Ambiguous {kind} directory under {raw_root}: "
            f"{[str(path.relative_to(raw_root)) for path in recursive_matches]}. "
            f"Pass --{kind.replace(' ', '-')} explicitly."
        )

    raise FileNotFoundError(
        f"Could not auto-discover a {kind} directory under {raw_root}. "
        f"Tried names: {preferred_names}. Pass --{kind.replace(' ', '-')} explicitly."
    )


def canonical_sample_id(path: Path) -> str:
    sample_id = MASK_SUFFIX_PATTERN.sub("", path.stem)
    return sample_id if sample_id else path.stem


def infer_prefix(sample_id: str) -> str:
    match = re.match(r"[A-Za-z]+", sample_id)
    if match:
        return match.group(0)
    return "other"


def collect_files(root: Path, suffixes: set[str]) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files)


def build_unique_id_map(paths: list[Path], kind: str) -> dict[str, Path]:
    mapping = {}
    duplicates = defaultdict(list)
    for path in paths:
        sample_id = canonical_sample_id(path)
        if sample_id in mapping:
            duplicates[sample_id].append(path)
            continue
        mapping[sample_id] = path

    if duplicates:
        lines = []
        for sample_id, dup_paths in sorted(duplicates.items()):
            all_paths = [mapping[sample_id], *dup_paths]
            lines.append(
                f"{sample_id}: {[str(path) for path in all_paths]}"
            )
        raise RuntimeError(
            f"Found duplicate {kind} sample IDs after canonicalization:\n" + "\n".join(lines)
        )
    return mapping


def compute_targets(total_count: int, ratios: dict[str, float]) -> dict[str, int]:
    raw_targets = {split: total_count * ratio for split, ratio in ratios.items()}
    floored = {split: math.floor(value) for split, value in raw_targets.items()}
    remainder = total_count - sum(floored.values())

    ranked_splits = sorted(
        SPLIT_NAMES,
        key=lambda split: (raw_targets[split] - floored[split], ratios[split]),
        reverse=True,
    )
    targets = dict(floored)
    for split in ranked_splits[:remainder]:
        targets[split] += 1
    return targets


def split_prefix_group(prefix_records, prefix_targets, seed):
    rng = np.random.default_rng(seed)
    decorated = [
        (row["fg_ratio"], float(rng.random()), row)
        for row in prefix_records
    ]
    sorted_records = [row for _, _, row in sorted(decorated, key=lambda item: (item[0], item[1]))]
    assignments = {split: [] for split in SPLIT_NAMES}

    for record in sorted_records:
        eligible_splits = [
            split for split in SPLIT_NAMES
            if len(assignments[split]) < prefix_targets[split]
        ]
        if not eligible_splits:
            raise RuntimeError("No eligible split left while assigning records.")

        chosen_split = min(
            eligible_splits,
            key=lambda split: (
                len(assignments[split]) / max(prefix_targets[split], 1),
                -prefix_targets[split],
                len(assignments[split]),
                split,
            ),
        )
        assignments[chosen_split].append(record)

    return assignments


def build_split_assignments(records, ratios, seed):
    global_targets = compute_targets(len(records), ratios)
    grouped_by_prefix = defaultdict(list)
    for record in records:
        grouped_by_prefix[record["prefix"]].append(record)

    final_assignments = {split: [] for split in SPLIT_NAMES}
    prefix_targets = {}
    for prefix, prefix_records in sorted(grouped_by_prefix.items()):
        prefix_target = compute_targets(len(prefix_records), ratios)
        prefix_targets[prefix] = prefix_target
        prefix_seed = seed + sum(ord(ch) for ch in prefix)
        prefix_assignments = split_prefix_group(prefix_records, prefix_target, prefix_seed)
        for split in SPLIT_NAMES:
            final_assignments[split].extend(prefix_assignments[split])

    for split in SPLIT_NAMES:
        final_assignments[split].sort(key=lambda row: row["sample_id"])

    actual_targets = {split: len(rows) for split, rows in final_assignments.items()}
    if actual_targets != global_targets:
        raise RuntimeError(
            f"Split counts do not match targets. targets={global_targets}, actual={actual_targets}"
        )

    return final_assignments, global_targets, prefix_targets


def summarize_rows(rows):
    prefix_counts = Counter(row["prefix"] for row in rows)
    fg_ratios = [row["fg_ratio"] for row in rows]
    heights = [row["height"] for row in rows]
    widths = [row["width"] for row in rows]
    return {
        "count": len(rows),
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "fg_ratio_mean": float(np.mean(fg_ratios)),
        "fg_ratio_median": float(np.median(fg_ratios)),
        "fg_ratio_min": float(np.min(fg_ratios)),
        "fg_ratio_max": float(np.max(fg_ratios)),
        "height_values": sorted(set(heights)),
        "width_values": sorted(set(widths)),
    }


def ensure_clean_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset root already exists: {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "masks").mkdir(parents=True, exist_ok=True)


def write_split_file(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['image_rel']} {row['mask_rel']}\n")


def export_dataset_assets(records, dataset_root: Path):
    for row in records:
        image_dst = dataset_root / row["image_rel"]
        mask_dst = dataset_root / row["mask_rel"]
        shutil.copy2(row["image_src"], image_dst)
        Image.fromarray(row["mask_binary"] * 255).save(mask_dst)


def load_records(raw_root: Path, images_dir: Path, masks_dir: Path):
    image_files = collect_files(images_dir, IMAGE_SUFFIXES)
    mask_files = collect_files(masks_dir, MASK_SUFFIXES)
    if not image_files:
        raise RuntimeError(f"No image files found under {images_dir}")
    if not mask_files:
        raise RuntimeError(f"No mask files found under {masks_dir}")

    image_map = build_unique_id_map(image_files, kind="image")
    mask_map = build_unique_id_map(mask_files, kind="mask")
    common_ids = sorted(set(image_map) & set(mask_map))
    if not common_ids:
        raise RuntimeError(
            f"No matched image/mask pairs found between {images_dir} and {masks_dir}"
        )

    records = []
    for sample_id in common_ids:
        image_src = image_map[sample_id]
        mask_src = mask_map[sample_id]
        image = Image.open(image_src).convert("RGB")
        mask_binary = np.array(Image.open(mask_src).convert("1"), dtype=np.uint8)
        height, width = mask_binary.shape
        records.append(
            {
                "sample_id": sample_id,
                "prefix": infer_prefix(sample_id),
                "image_src": image_src,
                "mask_src": mask_src,
                "image_src_rel": str(image_src.relative_to(raw_root)),
                "mask_src_rel": str(mask_src.relative_to(raw_root)),
                "image_rel": str(Path("images") / f"{sample_id}{image_src.suffix.lower()}"),
                "mask_rel": str(Path("masks") / f"{sample_id}.png"),
                "width": int(width),
                "height": int(height),
                "fg_ratio": float(mask_binary.mean()),
                "mask_binary": mask_binary,
            }
        )
        image.close()

    unmatched_images = sorted(set(image_map) - set(mask_map))
    unmatched_masks = sorted(set(mask_map) - set(image_map))
    return records, unmatched_images, unmatched_masks


def main():
    args = parse_args()
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    ratio_sum = sum(ratios.values())
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.6f}")

    raw_root = Path(args.raw_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    images_dir = resolve_input_dir(
        raw_root,
        args.images_dir,
        preferred_names=("images", "image", "imgs", "img", "rgb"),
        kind="images",
    )
    masks_dir = resolve_input_dir(
        raw_root,
        args.masks_dir,
        preferred_names=("masks", "mask", "labels", "label", "annotations", "annotation", "gt"),
        kind="masks",
    )

    records, unmatched_images, unmatched_masks = load_records(raw_root, images_dir, masks_dir)
    split_assignments, global_targets, prefix_targets = build_split_assignments(
        records,
        ratios,
        seed=args.seed,
    )

    ensure_clean_output_dir(dataset_root, overwrite=args.overwrite)
    export_dataset_assets(records, dataset_root)

    full_rows = sorted(records, key=lambda row: row["sample_id"])
    full_split_path = dataset_root / f"{args.full_split_name}.txt"
    write_split_file(full_split_path, full_rows)

    split_summaries = {}
    for split_name in SPLIT_NAMES:
        split_path = dataset_root / f"{split_name}.txt"
        write_split_file(split_path, split_assignments[split_name])
        split_summaries[split_name] = summarize_rows(split_assignments[split_name])
        summary = split_summaries[split_name]
        print(
            f"{split_name:5s} | count={summary['count']:4d} | "
            f"prefix={summary['prefix_counts']} | "
            f"fg_mean={summary['fg_ratio_mean']:.6f} | "
            f"fg_median={summary['fg_ratio_median']:.6f}"
        )

    manifest = {
        "dataset_name": "PaveCrack1300",
        "dataset_doi": "10.17632/8b27pdcxv7.1",
        "dataset_root": str(dataset_root),
        "raw_root": str(raw_root),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "seed": args.seed,
        "ratios": ratios,
        "full_split_name": args.full_split_name,
        "num_matched_pairs": len(records),
        "num_unmatched_images": len(unmatched_images),
        "num_unmatched_masks": len(unmatched_masks),
        "unmatched_image_ids": unmatched_images[:20],
        "unmatched_mask_ids": unmatched_masks[:20],
        "global_targets": global_targets,
        "prefix_targets": prefix_targets,
        "source_summary": summarize_rows(records),
        "split_summaries": split_summaries,
        "source_files": [
            {
                "sample_id": row["sample_id"],
                "image_src_rel": row["image_src_rel"],
                "mask_src_rel": row["mask_src_rel"],
                "width": row["width"],
                "height": row["height"],
                "fg_ratio": row["fg_ratio"],
            }
            for row in full_rows
        ],
    }
    manifest_path = dataset_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Raw root: {raw_root}")
    print(f"Resolved images dir: {images_dir}")
    print(f"Resolved masks dir: {masks_dir}")
    print(f"Prepared dataset root: {dataset_root}")
    print(f"Preserved full split: {full_split_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
