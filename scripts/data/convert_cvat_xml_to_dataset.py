import argparse
import json
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a CVAT for images 1.1 XML export with mask RLE annotations into "
            "the repository-native dataset format: images/, masks/, train.txt, val.txt, "
            "test.txt, crossdomain_all.txt, and split_manifest.json."
        )
    )
    parser.add_argument(
        "--annotations-xml",
        required=True,
        help="Path to the CVAT annotations.xml export.",
    )
    parser.add_argument(
        "--source-images-root",
        required=True,
        help="Directory that contains the original images referenced by the XML export.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Output dataset root in repository-native split-file format.",
    )
    parser.add_argument("--full-split-name", default="crossdomain_all")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink"),
        default="hardlink",
        help="How to materialize images into the prepared dataset root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing prepared dataset root.",
    )
    return parser.parse_args()


def infer_prefix(sample_id: str) -> str:
    prefix_chars = []
    for char in sample_id:
        if char.isalpha():
            prefix_chars.append(char)
            continue
        if prefix_chars:
            break
    return "".join(prefix_chars) if prefix_chars else "other"


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
    rng = random.Random(seed)
    decorated = [(row["fg_ratio"], rng.random(), row) for row in prefix_records]
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
    num_masks = [row["num_masks"] for row in rows]
    return {
        "count": len(rows),
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "fg_ratio_mean": float(np.mean(fg_ratios)),
        "fg_ratio_median": float(np.median(fg_ratios)),
        "fg_ratio_min": float(np.min(fg_ratios)),
        "fg_ratio_max": float(np.max(fg_ratios)),
        "num_masks_mean": float(np.mean(num_masks)),
        "num_masks_max": int(np.max(num_masks)),
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


def parse_rle_string(rle_text: str) -> list[int]:
    return [int(token.strip()) for token in rle_text.split(",") if token.strip()]


def decode_cvat_mask(mask_element: ET.Element) -> tuple[np.ndarray, int, int]:
    width = int(mask_element.attrib["width"])
    height = int(mask_element.attrib["height"])
    left = int(mask_element.attrib["left"])
    top = int(mask_element.attrib["top"])
    runs = parse_rle_string(mask_element.attrib["rle"])
    total_pixels = width * height
    if sum(runs) != total_pixels:
        raise ValueError(
            f"RLE run lengths do not match ROI size: sum={sum(runs)} vs {total_pixels}"
        )

    flat = np.zeros(total_pixels, dtype=np.uint8)
    cursor = 0
    value = 0
    for run in runs:
        if value == 1:
            flat[cursor: cursor + run] = 1
        cursor += run
        value ^= 1

    return flat.reshape(height, width), left, top


def materialize_image(src: Path, dst: Path, image_mode: str):
    if image_mode == "copy":
        shutil.copy2(src, dst)
        return

    if image_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    if image_mode == "symlink":
        relative_target = os.path.relpath(src, start=dst.parent)
        os.symlink(relative_target, dst)
        return

    raise ValueError(f"Unsupported image mode: {image_mode}")


def build_record(image_element: ET.Element, source_images_root: Path, dataset_root: Path, image_mode: str):
    image_name = image_element.attrib["name"]
    width = int(image_element.attrib["width"])
    height = int(image_element.attrib["height"])
    sample_id = Path(image_name).stem
    source_image = source_images_root / image_name
    if not source_image.exists():
        raise FileNotFoundError(f"Missing source image referenced by XML: {source_image}")

    with Image.open(source_image) as image:
        image_width, image_height = image.size
    if image_width != width or image_height != height:
        raise ValueError(
            f"Image size mismatch for {image_name}: XML={width}x{height}, file={image_width}x{image_height}"
        )

    full_mask = np.zeros((height, width), dtype=np.uint8)
    mask_elements = image_element.findall("mask")
    for mask_element in mask_elements:
        roi_mask, left, top = decode_cvat_mask(mask_element)
        roi_height, roi_width = roi_mask.shape
        bottom = top + roi_height
        right = left + roi_width
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(
                f"Mask ROI out of bounds for {image_name}: "
                f"left={left}, top={top}, width={roi_width}, height={roi_height}, "
                f"image={width}x{height}"
            )
        full_mask[top:bottom, left:right] |= roi_mask

    image_rel = Path("images") / f"{sample_id}{source_image.suffix.lower()}"
    mask_rel = Path("masks") / f"{sample_id}.png"
    image_dst = dataset_root / image_rel
    mask_dst = dataset_root / mask_rel

    materialize_image(source_image, image_dst, image_mode)
    Image.fromarray(full_mask * 255).save(mask_dst)

    return {
        "sample_id": sample_id,
        "prefix": infer_prefix(sample_id),
        "source_image_name": image_name,
        "source_image_path": str(source_image),
        "image_rel": str(image_rel),
        "mask_rel": str(mask_rel),
        "width": width,
        "height": height,
        "fg_ratio": float(full_mask.mean()),
        "num_masks": len(mask_elements),
        "has_foreground": bool(full_mask.any()),
    }


def export_records_from_xml(
    annotations_xml: Path,
    source_images_root: Path,
    dataset_root: Path,
    image_mode: str,
):
    root = ET.parse(annotations_xml).getroot()
    image_elements = root.findall("image")
    if not image_elements:
        raise ValueError(f"No <image> entries found in {annotations_xml}")

    task_size_text = root.findtext("./meta/task/size")
    label_names = [
        label_name.text.strip()
        for label_name in root.findall("./meta/task/labels/label/name")
        if label_name.text and label_name.text.strip()
    ]

    records = []
    for image_element in image_elements:
        records.append(build_record(image_element, source_images_root, dataset_root, image_mode))
    records.sort(key=lambda row: row["sample_id"])

    task_metadata = {
        "task_id": root.findtext("./meta/task/id"),
        "task_name": root.findtext("./meta/task/name"),
        "task_size": int(task_size_text) if task_size_text else None,
        "num_image_entries_in_xml": len(image_elements),
        "label_names": label_names,
        "xml_version": root.findtext("./version"),
        "created": root.findtext("./meta/task/created"),
        "updated": root.findtext("./meta/task/updated"),
        "dumped": root.findtext("./meta/dumped"),
    }
    return records, task_metadata


def write_split_file(path: Path, rows):
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

    annotations_xml = Path(args.annotations_xml).resolve()
    source_images_root = Path(args.source_images_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    if not annotations_xml.exists():
        raise FileNotFoundError(f"Annotations XML not found: {annotations_xml}")
    if not source_images_root.exists() or not source_images_root.is_dir():
        raise FileNotFoundError(f"Source images root not found: {source_images_root}")

    ensure_clean_output_dir(dataset_root, overwrite=args.overwrite)
    records, task_metadata = export_records_from_xml(
        annotations_xml,
        source_images_root,
        dataset_root,
        image_mode=args.image_mode,
    )

    split_assignments, global_targets, prefix_targets = build_split_assignments(
        records,
        ratios,
        seed=args.seed,
    )

    full_split_path = dataset_root / f"{args.full_split_name}.txt"
    write_split_file(full_split_path, records)

    split_summaries = {}
    for split_name in SPLIT_NAMES:
        split_path = dataset_root / f"{split_name}.txt"
        write_split_file(split_path, split_assignments[split_name])
        split_summaries[split_name] = summarize_rows(split_assignments[split_name])

    mask_count_histogram = Counter(row["num_masks"] for row in records)
    manifest = {
        "dataset_name": dataset_root.name,
        "dataset_root": str(dataset_root),
        "annotations_xml": str(annotations_xml),
        "source_images_root": str(source_images_root),
        "image_mode": args.image_mode,
        "seed": args.seed,
        "ratios": ratios,
        "full_split_name": args.full_split_name,
        "task_metadata": task_metadata,
        "num_records": len(records),
        "num_empty_annotation_entries": sum(1 for row in records if row["num_masks"] == 0),
        "num_zero_foreground_masks": sum(1 for row in records if not row["has_foreground"]),
        "mask_count_histogram": {str(key): value for key, value in sorted(mask_count_histogram.items())},
        "global_targets": global_targets,
        "prefix_targets": prefix_targets,
        "source_summary": summarize_rows(records),
        "split_summaries": split_summaries,
        "records": records,
    }
    manifest_path = dataset_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if task_metadata["task_size"] is not None and task_metadata["task_size"] != len(records):
        print(
            "NOTE: task metadata size differs from the number of image entries in XML: "
            f"task_size={task_metadata['task_size']}, xml_entries={len(records)}"
        )
    print(f"Prepared dataset root: {dataset_root}")
    print(f"Annotations XML: {annotations_xml}")
    print(f"Source images root: {source_images_root}")
    print(f"Records exported: {len(records)}")
    for split_name in SPLIT_NAMES:
        summary = split_summaries[split_name]
        print(
            f"{split_name:5s} | count={summary['count']:3d} | "
            f"prefix={summary['prefix_counts']} | "
            f"fg_mean={summary['fg_ratio_mean']:.6f} | "
            f"fg_median={summary['fg_ratio_median']:.6f}"
        )
    print(f"Preserved full split: {full_split_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
