import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible few-shot labeled subsets from the official UAV train split "
            "for B2 fine-tuning experiments."
        )
    )
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--source-train-split", default="train")
    parser.add_argument("--fractions", default="0.05,0.10,0.20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing few-shot split files.",
    )
    return parser.parse_args()


def parse_fraction_list(raw: str) -> list[float]:
    fractions = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Few-shot fraction must be in (0, 1], got {value}.")
        fractions.append(value)
    if not fractions:
        raise ValueError("At least one fraction must be provided.")
    return fractions


def load_records(dataset_root: Path, split_name: str):
    split_path = dataset_root / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing source split: {split_path}")

    records = []
    with split_path.open() as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Expected exactly 2 columns in {split_path} at line {line_number}, got: {stripped}"
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
        raise ValueError(f"No valid rows found in {split_path}")
    return records, split_path


def compute_target_count(total_count: int, fraction: float) -> int:
    return max(1, int(round(total_count * fraction)))


def compute_prefix_targets(records, fraction: float):
    total_count = len(records)
    target_count = compute_target_count(total_count, fraction)
    prefix_counts = Counter(row["prefix"] for row in records)

    base_targets = {prefix: 0 for prefix in prefix_counts}
    if target_count >= len(prefix_counts):
        for prefix in prefix_counts:
            if prefix_counts[prefix] > 0:
                base_targets[prefix] = 1
        target_count_remaining = target_count - sum(base_targets.values())
    else:
        target_count_remaining = target_count

    available_counts = {
        prefix: prefix_counts[prefix] - base_targets[prefix]
        for prefix in prefix_counts
    }
    available_total = sum(available_counts.values())
    if available_total <= 0:
        targets = dict(base_targets)
        if sum(targets.values()) != target_count:
            raise RuntimeError(
                f"Few-shot prefix targets do not sum to requested count. "
                f"requested={target_count}, actual={sum(targets.values())}"
            )
        return targets

    raw_targets = {
        prefix: available_counts[prefix] * (target_count_remaining / available_total)
        for prefix in prefix_counts
    }
    floored = {
        prefix: min(available_counts[prefix], math.floor(raw_targets[prefix]))
        for prefix in prefix_counts
    }
    remainder = target_count_remaining - sum(floored.values())

    ranked_prefixes = sorted(
        prefix_counts,
        key=lambda prefix: (raw_targets[prefix] - floored[prefix], available_counts[prefix]),
        reverse=True,
    )
    targets = {
        prefix: base_targets[prefix] + floored[prefix]
        for prefix in prefix_counts
    }
    for prefix in ranked_prefixes:
        if remainder <= 0:
            break
        if targets[prefix] >= prefix_counts[prefix]:
            continue
        targets[prefix] += 1
        remainder -= 1

    if sum(targets.values()) != target_count:
        raise RuntimeError(
            f"Few-shot prefix targets do not sum to requested count. "
            f"requested={target_count}, actual={sum(targets.values())}"
        )
    return targets


def select_evenly_spaced_rows(rows, target_count: int, rng: random.Random):
    if target_count >= len(rows):
        return sorted(rows, key=lambda row: row["sample_id"])

    decorated = [(row["fg_ratio"], rng.random(), row) for row in rows]
    sorted_rows = [row for _, _, row in sorted(decorated, key=lambda item: (item[0], item[1]))]

    if target_count == 1:
        picked = [sorted_rows[len(sorted_rows) // 2]]
    else:
        positions = np.linspace(0, len(sorted_rows) - 1, num=target_count)
        chosen_indices = []
        seen = set()
        for position in positions:
            idx = int(round(position))
            idx = max(0, min(len(sorted_rows) - 1, idx))
            while idx in seen and idx + 1 < len(sorted_rows):
                idx += 1
            while idx in seen and idx - 1 >= 0:
                idx -= 1
            seen.add(idx)
            chosen_indices.append(idx)
        picked = [sorted_rows[idx] for idx in sorted(chosen_indices)]

    return sorted(picked, key=lambda row: row["sample_id"])


def build_fewshot_subset(records, fraction: float, seed: int):
    prefix_targets = compute_prefix_targets(records, fraction)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["prefix"]].append(row)

    chosen = []
    for prefix, rows in sorted(grouped.items()):
        rng = random.Random(seed + sum(ord(ch) for ch in prefix) + int(fraction * 1000))
        chosen.extend(select_evenly_spaced_rows(rows, prefix_targets[prefix], rng))

    chosen.sort(key=lambda row: row["sample_id"])
    target_count = compute_target_count(len(records), fraction)
    if len(chosen) != target_count:
        raise RuntimeError(
            f"Few-shot subset size mismatch for fraction={fraction}. "
            f"expected={target_count}, actual={len(chosen)}"
        )
    return chosen, prefix_targets


def summarize_rows(rows):
    prefix_counts = Counter(row["prefix"] for row in rows)
    fg_ratios = [row["fg_ratio"] for row in rows]
    return {
        "count": len(rows),
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "fg_ratio_mean": float(np.mean(fg_ratios)),
        "fg_ratio_median": float(np.median(fg_ratios)),
        "fg_ratio_min": float(np.min(fg_ratios)),
        "fg_ratio_max": float(np.max(fg_ratios)),
        "sample_ids": [row["sample_id"] for row in rows],
    }


def fraction_tag(fraction: float) -> str:
    percentage = int(round(fraction * 100))
    return f"fs{percentage:02d}"


def write_split_file(path: Path, rows, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing split without --overwrite: {path}")
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['image_rel']} {row['mask_rel']}\n")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    fractions = parse_fraction_list(args.fractions)
    records, source_split_path = load_records(dataset_root, args.source_train_split)

    manifest = {
        "dataset_root": str(dataset_root.resolve()),
        "source_train_split": args.source_train_split,
        "seed": args.seed,
        "fractions": fractions,
        "source_summary": summarize_rows(records),
        "subsets": {},
    }

    for fraction in fractions:
        tag = fraction_tag(fraction)
        rows, prefix_targets = build_fewshot_subset(records, fraction, args.seed)
        split_name = f"{args.source_train_split}_{tag}_seed{args.seed}"
        split_path = dataset_root / f"{split_name}.txt"
        write_split_file(split_path, rows, overwrite=args.overwrite)
        manifest["subsets"][split_name] = {
            "fraction": fraction,
            "prefix_targets": prefix_targets,
            "summary": summarize_rows(rows),
            "path": str(split_path.resolve()),
        }
        summary = manifest["subsets"][split_name]["summary"]
        print(
            f"{split_name:24s} | count={summary['count']:3d} | "
            f"prefix={summary['prefix_counts']} | "
            f"fg_mean={summary['fg_ratio_mean']:.6f} | "
            f"fg_median={summary['fg_ratio_median']:.6f}"
        )

    manifest_path = dataset_root / f"fewshot_manifest_seed{args.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Source train split: {source_split_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
