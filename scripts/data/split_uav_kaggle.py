import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible train/val/test splits for UAV_Crack_Segmentation_Kaggle "
            "while preserving the original all-sample cross-domain list."
        )
    )
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--full-split-name", default="crossdomain_all")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing train/val/test files after backing up the source split.",
    )
    return parser.parse_args()


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


def load_records(dataset_root: Path, source_split_name: str):
    source_path = dataset_root / f"{source_split_name}.txt"
    if not source_path.exists():
        raise FileNotFoundError(f"Source split not found: {source_path}")

    records = []
    with source_path.open() as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Expected exactly 2 columns in {source_path} at line {line_number}, "
                    f"got: {stripped}"
                )

            image_rel, mask_rel = parts
            image_path = dataset_root / image_rel
            mask_path = dataset_root / mask_rel
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask file: {mask_path}")

            sample_id = image_path.stem
            if sample_id.startswith("DJI"):
                prefix = "DJI"
            elif sample_id.startswith("slide"):
                prefix = "slide"
            else:
                prefix = "other"

            mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)
            fg_ratio = float(mask.mean())
            records.append(
                {
                    "sample_id": sample_id,
                    "prefix": prefix,
                    "image_rel": image_rel,
                    "mask_rel": mask_rel,
                    "fg_ratio": fg_ratio,
                }
            )

    if not records:
        raise ValueError(f"No valid rows found in {source_path}")

    return records, source_path


def split_prefix_group(prefix_records, prefix_targets, seed):
    rng = random.Random(seed)
    records_with_tiebreak = [
        (row["fg_ratio"], rng.random(), row)
        for row in prefix_records
    ]
    sorted_records = [row for _, _, row in sorted(records_with_tiebreak, key=lambda item: (item[0], item[1]))]
    assignments = {split: [] for split in SPLIT_NAMES}

    for record in sorted_records:
        eligible_splits = [split for split in SPLIT_NAMES if len(assignments[split]) < prefix_targets[split]]
        if not eligible_splits:
            raise RuntimeError("No eligible split left while assigning prefix group.")

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
        prefix_assignments = split_prefix_group(prefix_records, prefix_target, seed=prefix_seed)
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


def summarize_split(rows):
    prefix_counts = Counter(row["prefix"] for row in rows)
    fg_ratios = [row["fg_ratio"] for row in rows]
    return {
        "count": len(rows),
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "fg_ratio_mean": float(np.mean(fg_ratios)),
        "fg_ratio_median": float(np.median(fg_ratios)),
        "fg_ratio_min": float(np.min(fg_ratios)),
        "fg_ratio_max": float(np.max(fg_ratios)),
    }


def write_split_file(path: Path, rows, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing split file without --overwrite: {path}"
        )

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['image_rel']} {row['mask_rel']}\n")


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

    dataset_root = Path(args.dataset_root)
    records, source_path = load_records(dataset_root, args.source_split)
    full_split_path = dataset_root / f"{args.full_split_name}.txt"
    manifest_path = dataset_root / "split_manifest.json"

    full_rows = sorted(records, key=lambda row: row["sample_id"])
    if full_split_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing full split file without --overwrite: {full_split_path}"
        )

    write_split_file(full_split_path, full_rows, overwrite=args.overwrite)

    split_assignments, global_targets, prefix_targets = build_split_assignments(
        records,
        ratios,
        seed=args.seed,
    )

    for split_name in SPLIT_NAMES:
        split_path = dataset_root / f"{split_name}.txt"
        overwrite_allowed = args.overwrite or split_path == source_path
        write_split_file(split_path, split_assignments[split_name], overwrite=overwrite_allowed)

    manifest = {
        "dataset_root": str(dataset_root.resolve()),
        "source_split": args.source_split,
        "full_split_name": args.full_split_name,
        "seed": args.seed,
        "ratios": ratios,
        "targets": global_targets,
        "prefix_targets": prefix_targets,
        "summary": {
            split_name: summarize_split(rows)
            for split_name, rows in split_assignments.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Source split: {source_path}")
    print(f"Preserved full cross-domain list at: {full_split_path}")
    for split_name in SPLIT_NAMES:
        summary = manifest["summary"][split_name]
        print(
            f"{split_name:5s} | count={summary['count']:3d} | "
            f"prefix={summary['prefix_counts']} | "
            f"fg_mean={summary['fg_ratio_mean']:.6f} | "
            f"fg_median={summary['fg_ratio_median']:.6f}"
        )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
