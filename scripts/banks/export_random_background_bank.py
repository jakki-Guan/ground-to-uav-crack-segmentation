import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crack_detection.dataset import CrackDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a random all-background bank matched to an existing mined bank. "
            "By default, each random crop matches the reference crop size and source "
            "image, while avoiding overlap with the reference crop region."
        )
    )
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--reference-bank-root",
        required=True,
        help="Reference bank root that contains manifest.jsonl and train.txt.",
    )
    parser.add_argument(
        "--reference-bank-label",
        default=None,
        help="Optional logical label for the reference bank. Defaults to parent__name.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output dataset root. Defaults to generated/random_background_banks/"
            "<reference_label>__random_bg."
        ),
    )
    parser.add_argument(
        "--source-image-mode",
        choices=["same", "global"],
        default="same",
        help=(
            "Whether to sample the random control crop from the same UAV source image "
            "as each reference entry or from the full split."
        ),
    )
    parser.add_argument(
        "--strict-source-match",
        action="store_true",
        help="Fail instead of falling back to the global split when same-image sampling fails.",
    )
    parser.add_argument(
        "--allow-reference-overlap",
        action="store_true",
        help="Allow a random crop to overlap the reference crop region when sampling from the same image.",
    )
    parser.add_argument(
        "--max-attempts-per-entry",
        type=int,
        default=400,
        help="Maximum random trials before trying a fallback strategy for one entry.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output root if it already exists.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def ensure_clean_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {path}. Pass --overwrite to replace it."
            )
        for child in path.iterdir():
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
        path.rmdir()
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "masks").mkdir(parents=True, exist_ok=True)


def build_sample_id_to_index(dataset: CrackDataset) -> dict[str, int]:
    mapping = {}
    for idx, (img_path, _) in enumerate(dataset.samples):
        sample_id = Path(img_path).stem
        if sample_id in mapping:
            raise ValueError(f"Duplicate sample id in split manifest: {sample_id}")
        mapping[sample_id] = idx
    return mapping


def build_integral_mask(mask: np.ndarray) -> np.ndarray:
    integral = mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant")


def area_sum(integral: np.ndarray, left: int, top: int, side: int) -> int:
    right = left + side
    bottom = top + side
    return int(
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )


def boxes_overlap(a_left: int, a_top: int, a_side: int, b_left: int, b_top: int, b_side: int) -> bool:
    a_right = a_left + a_side
    a_bottom = a_top + a_side
    b_right = b_left + b_side
    b_bottom = b_top + b_side
    return not (
        a_right <= b_left
        or b_right <= a_left
        or a_bottom <= b_top
        or b_bottom <= a_top
    )


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def summarize_entries(entries: list[dict]) -> dict:
    if not entries:
        return {"num_crops": 0, "unique_source_images": 0}

    crop_sizes = [int(entry["crop_size"]) for entry in entries]
    same_image_flags = [int(entry["sampled_from_same_source_image"]) for entry in entries]
    return {
        "num_crops": len(entries),
        "unique_source_images": len({entry["source_sample_id"] for entry in entries}),
        "crop_size_mean": mean(crop_sizes),
        "crop_size_median": median(crop_sizes),
        "same_source_match_fraction": mean(same_image_flags),
    }


def load_raw_cache(dataset: CrackDataset) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    images = []
    masks = []
    integral_masks = []
    for idx in range(len(dataset)):
        image, mask = dataset.get_raw(idx)
        images.append(image)
        masks.append(mask)
        integral_masks.append(build_integral_mask(mask))
    return images, masks, integral_masks


def sample_random_crop(
    rng: random.Random,
    image_shape: tuple[int, int, int],
    integral_mask: np.ndarray,
    side: int,
    reference_crop: tuple[int, int, int] | None,
    used_locations: set[tuple[int, int, int, int]],
    max_attempts: int,
    allow_reference_overlap: bool,
    image_index: int,
):
    height, width = image_shape[:2]
    if side > width or side > height:
        return None

    max_left = width - side
    max_top = height - side
    for _ in range(max_attempts):
        left = rng.randint(0, max_left) if max_left > 0 else 0
        top = rng.randint(0, max_top) if max_top > 0 else 0

        if area_sum(integral_mask, left, top, side) > 0:
            continue
        if reference_crop is not None and not allow_reference_overlap:
            ref_left, ref_top, ref_side = reference_crop
            if boxes_overlap(left, top, side, ref_left, ref_top, ref_side):
                continue

        location_key = (image_index, left, top, side)
        if location_key in used_locations:
            continue
        used_locations.add(location_key)
        return left, top

    return None


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    reference_bank_root = Path(args.reference_bank_root).resolve()
    manifest_path = reference_bank_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing reference manifest: {manifest_path}")

    reference_entries = load_jsonl(manifest_path)
    if not reference_entries:
        raise ValueError(f"Reference bank is empty: {reference_bank_root}")

    reference_label = args.reference_bank_label or (
        f"{reference_bank_root.parent.name}__{reference_bank_root.name}"
    )
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (Path("generated") / "random_background_banks" / f"{reference_label}__random_bg").resolve()
    )
    ensure_clean_output_dir(output_root, overwrite=args.overwrite)

    dataset = CrackDataset(args.dataset_root, split=args.split, img_size=360)
    sample_id_to_index = build_sample_id_to_index(dataset)
    images, _, integral_masks = load_raw_cache(dataset)
    global_indices = list(range(len(dataset)))
    used_locations: set[tuple[int, int, int, int]] = set()

    print(f"Reference bank root: {reference_bank_root}")
    print(f"Output root: {output_root}")
    print(f"Sampling split: {args.split} ({len(dataset)} source images)")
    print(
        "Matching config: "
        f"entries={len(reference_entries)}, "
        f"source_image_mode={args.source_image_mode}, "
        f"strict_source_match={args.strict_source_match}, "
        f"allow_reference_overlap={args.allow_reference_overlap}, "
        f"seed={args.seed}"
    )

    selected_entries = []
    fallback_count = 0
    same_source_count = 0

    for rank, ref_entry in enumerate(reference_entries):
        ref_sample_id = ref_entry["source_sample_id"]
        ref_side = int(ref_entry["crop_size"])
        ref_left = int(ref_entry.get("crop_left", 0))
        ref_top = int(ref_entry.get("crop_top", 0))
        reference_crop = (ref_left, ref_top, ref_side)

        if ref_sample_id not in sample_id_to_index:
            raise KeyError(
                f"Reference sample id not found in dataset split '{args.split}': {ref_sample_id}"
            )

        preferred_index = sample_id_to_index[ref_sample_id]
        candidate_indices = (
            [preferred_index]
            if args.source_image_mode == "same"
            else global_indices
        )

        chosen_index = None
        chosen_left = None
        chosen_top = None

        for image_index in candidate_indices:
            ref_crop_for_image = reference_crop if image_index == preferred_index else None
            sampled = sample_random_crop(
                rng=rng,
                image_shape=images[image_index].shape,
                integral_mask=integral_masks[image_index],
                side=ref_side,
                reference_crop=ref_crop_for_image,
                used_locations=used_locations,
                max_attempts=args.max_attempts_per_entry,
                allow_reference_overlap=args.allow_reference_overlap,
                image_index=image_index,
            )
            if sampled is not None:
                chosen_index = image_index
                chosen_left, chosen_top = sampled
                break

        if chosen_index is None and args.source_image_mode == "same" and not args.strict_source_match:
            fallback_indices = [idx for idx in global_indices if idx != preferred_index]
            rng.shuffle(fallback_indices)
            for image_index in fallback_indices:
                sampled = sample_random_crop(
                    rng=rng,
                    image_shape=images[image_index].shape,
                    integral_mask=integral_masks[image_index],
                    side=ref_side,
                    reference_crop=None,
                    used_locations=used_locations,
                    max_attempts=args.max_attempts_per_entry,
                    allow_reference_overlap=True,
                    image_index=image_index,
                )
                if sampled is not None:
                    chosen_index = image_index
                    chosen_left, chosen_top = sampled
                    fallback_count += 1
                    break

        if chosen_index is None:
            raise RuntimeError(
                "Failed to sample a valid random background crop for "
                f"reference rank {rank} ({ref_sample_id}, side={ref_side})."
            )

        sampled_from_same_source_image = chosen_index == preferred_index
        same_source_count += int(sampled_from_same_source_image)
        source_image = images[chosen_index]
        crop_image = source_image[
            chosen_top: chosen_top + ref_side,
            chosen_left: chosen_left + ref_side,
        ]
        crop_mask = np.zeros((ref_side, ref_side), dtype=np.uint8)

        chosen_sample_id = Path(dataset.samples[chosen_index][0]).stem
        base_name = (
            f"{rank:04d}_{chosen_sample_id}"
            f"_x{chosen_left}_y{chosen_top}_s{ref_side}.png"
        )
        image_rel = Path("images") / base_name
        mask_rel = Path("masks") / base_name
        Image.fromarray(crop_image).save(output_root / image_rel)
        Image.fromarray(crop_mask * 255).save(output_root / mask_rel)

        selected_entries.append(
            {
                "reference_rank": rank,
                "reference_image_rel": ref_entry.get("image_rel"),
                "reference_mask_rel": ref_entry.get("mask_rel"),
                "reference_source_sample_id": ref_sample_id,
                "reference_crop_left": ref_left,
                "reference_crop_top": ref_top,
                "reference_crop_size": ref_side,
                "source_index": chosen_index,
                "source_sample_id": chosen_sample_id,
                "crop_left": chosen_left,
                "crop_top": chosen_top,
                "crop_size": ref_side,
                "crop_gt_foreground_pixels": 0,
                "sampled_from_same_source_image": sampled_from_same_source_image,
                "image_rel": str(image_rel),
                "mask_rel": str(mask_rel),
            }
        )

    train_txt_path = output_root / "train.txt"
    with train_txt_path.open("w", encoding="utf-8") as f:
        for entry in selected_entries:
            f.write(f"{entry['image_rel']} {entry['mask_rel']}\n")

    manifest_jsonl_path = output_root / "manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as f:
        for entry in selected_entries:
            f.write(json.dumps(entry) + "\n")

    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": args.split,
        "reference_bank_root": str(reference_bank_root),
        "reference_bank_label": reference_label,
        "output_root": str(output_root),
        "source_image_mode": args.source_image_mode,
        "strict_source_match": args.strict_source_match,
        "allow_reference_overlap": args.allow_reference_overlap,
        "seed": args.seed,
        "max_attempts_per_entry": args.max_attempts_per_entry,
        "reference_entry_count": len(reference_entries),
        "same_source_match_count": same_source_count,
        "cross_image_fallback_count": fallback_count,
        "selected_summary": summarize_entries(selected_entries),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Exported {len(selected_entries)} random background crops "
        f"with {same_source_count}/{len(selected_entries)} same-image matches."
    )
    if fallback_count > 0:
        print(f"Cross-image fallbacks used: {fallback_count}")
    print(f"Dataset list written to: {train_txt_path}")
    print(f"Manifest written to: {manifest_jsonl_path}")
    print(f"Summary written to: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
