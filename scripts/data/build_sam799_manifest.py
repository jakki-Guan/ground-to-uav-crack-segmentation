import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a paper-facing SAM799 external-test manifest and optionally create "
            "an `external_test.txt` split alias that points at the full fixed set."
        )
    )
    parser.add_argument("--dataset-root", default="SAM799_CVAT")
    parser.add_argument("--source-split", default="crossdomain_all")
    parser.add_argument("--external-split-name", default="external_test")
    parser.add_argument(
        "--output-csv",
        default="results/external_sam799_cvat_patchwise/sam799_manifest.csv",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_split_rows(dataset_root: Path, split_name: str) -> list[tuple[str, str]]:
    split_path = dataset_root / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    pairs: list[tuple[str, str]] = []
    with split_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Expected exactly two columns in {split_path} line {line_number}, got: {stripped}"
                )
            pairs.append((parts[0], parts[1]))

    if not pairs:
        raise ValueError(f"No image/mask rows found in {split_path}")
    return pairs


def ensure_external_split(dataset_root: Path, split_name: str, pairs: list[tuple[str, str]]):
    split_path = dataset_root / f"{split_name}.txt"
    with split_path.open("w", encoding="utf-8") as f:
        for image_rel, mask_rel in pairs:
            f.write(f"{image_rel} {mask_rel}\n")
    return split_path


def build_manifest_rows(dataset_root: Path, pairs: list[tuple[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_rel, mask_rel in pairs:
        image_path = dataset_root / image_rel
        mask_path = dataset_root / mask_rel
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")

        with Image.open(image_path) as image:
            width, height = image.size
        mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)
        if mask.shape != (height, width):
            raise ValueError(
                f"Image/mask size mismatch for {image_rel}: image={width}x{height}, "
                f"mask={mask.shape[1]}x{mask.shape[0]}"
            )

        mask_values = set(np.unique(mask).tolist())
        if not mask_values.issubset({0, 1}):
            raise ValueError(f"Mask is not binary for {mask_rel}: values={sorted(mask_values)}")

        fg_pixels = int(mask.sum())
        total_pixels = int(mask.size)
        sample_id = Path(image_rel).stem
        rows.append(
            {
                "image_id": sample_id,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "width": width,
                "height": height,
                "has_crack": int(fg_pixels > 0),
                "fg_pixels": fg_pixels,
                "total_pixels": total_pixels,
                "fg_ratio": float(fg_pixels / total_pixels),
                "split_role": "external_test",
            }
        )

    rows.sort(key=lambda row: str(row["image_id"]))
    return rows


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    pairs = load_split_rows(dataset_root, args.source_split)
    external_split_path = ensure_external_split(
        dataset_root=dataset_root,
        split_name=args.external_split_name,
        pairs=pairs,
    )
    manifest_rows = build_manifest_rows(dataset_root=dataset_root, pairs=pairs)
    output_csv = Path(args.output_csv).resolve()
    write_csv(output_csv, manifest_rows)

    num_images = len(manifest_rows)
    num_positive = sum(int(row["has_crack"]) for row in manifest_rows)
    num_empty = num_images - num_positive
    mean_fg_ratio = float(np.mean([float(row["fg_ratio"]) for row in manifest_rows]))

    print(f"Dataset root: {dataset_root}")
    print(f"Source split: {args.source_split}")
    print(f"External split written to: {external_split_path}")
    print(f"Manifest written to: {output_csv}")
    print(
        f"Images: {num_images} | positive: {num_positive} | empty negative: {num_empty} | "
        f"mean fg ratio: {mean_fg_ratio:.6f}"
    )


if __name__ == "__main__":
    main()
