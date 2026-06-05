import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crack_detection.dataset import CrackDataset
from crack_detection.model import SEGFORMER_B2_MODEL_NAME, get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Mine hard-negative background crops from UAV imagery using high-confidence "
            "false-positive regions from a frozen segmentation model."
        )
    )
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="segformer-b2")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--pretrained-model-name", default=SEGFORMER_B2_MODEL_NAME)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--temperature-json",
        default=None,
        help=(
            "Optional temperature-scaling JSON from scripts/banks/fit_temperature.py. "
            "When provided, hard-negative mining uses sigmoid(logits / temperature)."
        ),
    )
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--component-threshold", type=float, default=0.8)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--min-component-mean-prob", type=float, default=0.85)
    parser.add_argument("--min-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--max-fill-ratio", type=float, default=1.0)
    parser.add_argument(
        "--allow-crop-gt-foreground",
        action="store_true",
        help=(
            "Allow a mined crop to contain some ground-truth crack pixels. "
            "By default, the bank keeps only pure-background crops because it is "
            "intended for all-background supervision."
        ),
    )
    parser.add_argument(
        "--crop-context-scale",
        type=float,
        default=1.6,
        help="Expand the component bounding box by this factor before taking a square crop.",
    )
    parser.add_argument("--min-crop-size", type=int, default=128)
    parser.add_argument("--max-crop-size", type=int, default=256)
    parser.add_argument("--max-crops-per-image", type=int, default=3)
    parser.add_argument("--max-crops-total", type=int, default=300)
    parser.add_argument(
        "--output-root",
        default="generated/uav_hard_negatives_segformer_b2_train_thr080",
        help=(
            "Output dataset root. The script will create images/, masks/, train.txt, "
            "manifest.jsonl, and summary.json so the result can be loaded as a dataset."
        ),
    )
    return parser.parse_args()


def load_temperature_from_json(temperature_json: str | None) -> float | None:
    if not temperature_json:
        return None

    payload = json.loads(Path(temperature_json).read_text(encoding="utf-8"))
    if "fitted_temperature" not in payload:
        raise KeyError(
            f"Temperature JSON does not contain 'fitted_temperature': {temperature_json}"
        )
    temperature = float(payload["fitted_temperature"])
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature} from {temperature_json}")
    return temperature


def square_crop_bounds(center_x, center_y, side, width, height):
    side = min(side, width, height)
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return left, top, side


def component_to_crop(stats_row, image_shape, context_scale, min_crop_size, max_crop_size):
    x, y, width, height, _ = [int(value) for value in stats_row]
    raw_h, raw_w = image_shape[:2]
    center_x = x + width / 2
    center_y = y + height / 2

    side = int(round(max(width, height) * context_scale))
    side = max(side, min_crop_size)
    side = min(side, max_crop_size, raw_h, raw_w)
    return square_crop_bounds(center_x, center_y, side, raw_w, raw_h)


def summarize_bank(entries):
    if not entries:
        return {
            "num_crops": 0,
            "unique_source_images": 0,
        }

    values = {
        "component_area": [entry["component_area"] for entry in entries],
        "component_mean_prob": [entry["component_mean_prob"] for entry in entries],
        "crop_size": [entry["crop_size"] for entry in entries],
    }
    return {
        "num_crops": len(entries),
        "unique_source_images": len({entry["source_sample_id"] for entry in entries}),
        "component_area_mean": float(np.mean(values["component_area"])),
        "component_area_median": float(np.median(values["component_area"])),
        "component_mean_prob_mean": float(np.mean(values["component_mean_prob"])),
        "component_mean_prob_median": float(np.median(values["component_mean_prob"])),
        "crop_size_mean": float(np.mean(values["crop_size"])),
        "crop_size_median": float(np.median(values["crop_size"])),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = load_temperature_from_json(args.temperature_json)
    output_root = Path(args.output_root)
    images_dir = output_root / "images"
    masks_dir = output_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    dataset = CrackDataset(args.dataset_root, split=args.split, img_size=args.img_size)
    model = get_model(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
        pretrained_model_name=args.pretrained_model_name,
    )
    state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Mining split: {args.split} ({len(dataset)} samples)")
    print(f"Output root: {output_root}")
    print(
        "Mining config: "
        f"thr={args.component_threshold}, "
        f"min_area={args.min_component_area}, "
        f"min_mean_prob={args.min_component_mean_prob}, "
        f"max_crops_per_image={args.max_crops_per_image}, "
        f"max_crops_total={args.max_crops_total}"
    )
    if temperature is not None:
        print(
            "Calibration config: "
            f"temperature={temperature:.6f} "
            f"(prob = sigmoid(logits / temperature))"
        )

    candidate_entries = []
    for idx in range(len(dataset)):
        image_t, _ = dataset[idx]
        raw_image, raw_mask = dataset.get_raw(idx)
        sample_id = Path(dataset.samples[idx][0]).stem

        with torch.no_grad():
            logits = model(image_t.unsqueeze(0).to(device))
            if temperature is not None:
                logits = logits / temperature
            prob_map_small = torch.sigmoid(logits)[0, 0].cpu().numpy()

        prob_map = cv2.resize(
            prob_map_small,
            (raw_image.shape[1], raw_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        binary_mask = (prob_map > args.component_threshold).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        per_image_entries = []
        for component_id in range(1, component_count):
            x, y, width, height, area = [int(value) for value in stats[component_id]]
            if area < args.min_component_area:
                continue

            fill_ratio = float(area) / float(max(width * height, 1))
            if fill_ratio > args.max_fill_ratio:
                continue

            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            if aspect_ratio < args.min_aspect_ratio:
                continue

            component_mask = labels == component_id
            overlap_pixels = int((raw_mask[component_mask] > 0).sum())
            if overlap_pixels > 0:
                continue

            component_mean_prob = float(prob_map[component_mask].mean())
            if component_mean_prob < args.min_component_mean_prob:
                continue

            left, top, side = component_to_crop(
                stats[component_id],
                raw_image.shape,
                context_scale=args.crop_context_scale,
                min_crop_size=args.min_crop_size,
                max_crop_size=args.max_crop_size,
            )
            crop_image = raw_image[top: top + side, left: left + side]
            crop_gt_mask = raw_mask[top: top + side, left: left + side]
            if not args.allow_crop_gt_foreground and int(crop_gt_mask.sum()) > 0:
                continue

            score = component_mean_prob * np.sqrt(area)
            per_image_entries.append(
                {
                    "source_index": idx,
                    "source_sample_id": sample_id,
                    "component_id": component_id,
                    "component_area": area,
                    "component_bbox_xywh": [x, y, width, height],
                    "component_mean_prob": component_mean_prob,
                    "component_aspect_ratio": float(aspect_ratio),
                    "component_fill_ratio": float(fill_ratio),
                    "crop_left": int(left),
                    "crop_top": int(top),
                    "crop_size": int(side),
                    "crop_gt_foreground_pixels": int(crop_gt_mask.sum()),
                    "score": float(score),
                }
            )

        per_image_entries.sort(key=lambda entry: entry["score"], reverse=True)
        candidate_entries.extend(per_image_entries[: args.max_crops_per_image])

    candidate_entries.sort(key=lambda entry: entry["score"], reverse=True)
    selected_entries = candidate_entries[: args.max_crops_total]

    for rank, entry in enumerate(selected_entries):
        raw_image, _ = dataset.get_raw(entry["source_index"])
        side = entry["crop_size"]
        left = entry["crop_left"]
        top = entry["crop_top"]
        crop_image = raw_image[top: top + side, left: left + side]
        crop_mask = np.zeros((side, side), dtype=np.uint8)

        base_name = (
            f"{rank:04d}_{entry['source_sample_id']}"
            f"_c{entry['component_id']}"
            f"_s{side}.png"
        )
        image_rel = Path("images") / base_name
        mask_rel = Path("masks") / base_name
        Image.fromarray(crop_image).save(output_root / image_rel)
        Image.fromarray(crop_mask * 255).save(output_root / mask_rel)

        entry["image_rel"] = str(image_rel)
        entry["mask_rel"] = str(mask_rel)

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
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "temperature_json": (
            str(Path(args.temperature_json).resolve()) if args.temperature_json else None
        ),
        "temperature": temperature,
        "model_name": args.model_name,
        "img_size": args.img_size,
        "component_threshold": args.component_threshold,
        "min_component_area": args.min_component_area,
        "min_component_mean_prob": args.min_component_mean_prob,
        "min_aspect_ratio": args.min_aspect_ratio,
        "max_fill_ratio": args.max_fill_ratio,
        "allow_crop_gt_foreground": args.allow_crop_gt_foreground,
        "crop_context_scale": args.crop_context_scale,
        "min_crop_size": args.min_crop_size,
        "max_crop_size": args.max_crop_size,
        "max_crops_per_image": args.max_crops_per_image,
        "max_crops_total": args.max_crops_total,
        "candidate_count_before_global_cap": len(candidate_entries),
        "selected_summary": summarize_bank(selected_entries),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Mined {len(selected_entries)} hard-negative crops "
        f"from {summary['selected_summary']['unique_source_images']} source images."
    )
    print(f"Dataset list written to: {train_txt_path}")
    print(f"Manifest written to: {manifest_jsonl_path}")
    print(f"Summary written to: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
