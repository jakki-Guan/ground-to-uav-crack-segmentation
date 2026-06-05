import argparse
import copy
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MmsegStageSpec:
    key: str
    label: str
    checkpoint_path: str
    threshold: float
    config_path: str
    mmseg_root: str
    model_name: str
    postprocess_min_area: int = 0
    postprocess_max_fill_ratio: float = 1.0
    postprocess_min_aspect_ratio: float = 1.0
    postprocess_max_components: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-protocol patch-based external evaluation on SAM799-CVAT "
            "for a single DAFormer/HRDA-style MMSeg checkpoint."
        )
    )
    parser.add_argument("--dataset-root", default="SAM799_CVAT")
    parser.add_argument("--split", default="external_test")
    parser.add_argument(
        "--manifest-csv",
        default="results/external_sam799_cvat_patchwise/sam799_manifest.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-key", required=True)
    parser.add_argument("--stage-label", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mmseg-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--model-input-size", type=int, default=360)
    parser.add_argument("--patch-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-probability-maps", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument("--save-limit", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    image_rel: str
    mask_rel: str
    image_path: str
    mask_path: str
    width: int
    height: int
    fg_pixels: int
    total_pixels: int
    fg_ratio: float
    empty_gt: bool


@dataclass(frozen=True)
class PatchPlan:
    image_id: str
    width: int
    height: int
    patch_size: int
    stride: int
    num_patches: int
    coverage_ok: bool
    min_coverage: int
    max_coverage: int
    patches: tuple[tuple[int, int, int, int], ...]


def add_mmseg_to_path(mmseg_root: str):
    root = str(Path(mmseg_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def import_mmseg_modules():
    import mmcv
    from mmcv.parallel import MMDataParallel
    from mmseg.models import build_segmentor

    return mmcv, MMDataParallel, build_segmentor


def load_student_state_dict(model, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    filtered = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            filtered[key[len("model."):]] = value
        elif not key.startswith("ema_") and not key.startswith("imnet_"):
            filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        "Loaded checkpoint: {} (student tensors={}, missing={}, unexpected={})".format(
            checkpoint_path, len(filtered), len(missing), len(unexpected)
        )
    )
    if missing:
        print("First missing keys: {}".format(", ".join(missing[:10])))
    if unexpected:
        print("First unexpected keys: {}".format(", ".join(unexpected[:10])))


def init_model(cfg, checkpoint_path: str, device: torch.device, build_segmentor, MMDataParallel):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_student_state_dict(model, checkpoint_path)
    model = model.to(device)
    if device.type == "cuda":
        model = MMDataParallel(model, device_ids=[device.index or 0])
    model.eval()
    return model


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory_stats(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_gpu_peak_memory_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return math.nan, math.nan
    allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return float(allocated_mb), float(reserved_mb)


def _read_proc_status_value_mb(field_name: str) -> float:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return math.nan

    prefix = f"{field_name}:"
    with status_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.startswith(prefix):
                continue
            parts = line.split()
            if len(parts) < 3:
                return math.nan
            try:
                value_kib = float(parts[1])
            except ValueError:
                return math.nan
            return float(value_kib / 1024.0)
    return math.nan


def get_process_memory_mb() -> tuple[float, float]:
    return _read_proc_status_value_mb("VmRSS"), _read_proc_status_value_mb("VmHWM")


def write_csv(path: Path, rows: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_split_pairs(dataset_root: Path, split_name: str) -> list[tuple[str, str]]:
    split_path = dataset_root / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    pairs = []
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
        raise ValueError(f"No samples found in {split_path}")
    return pairs


def load_samples(dataset_root: Path, split_name: str, max_images: int | None = None) -> list[SampleRecord]:
    pairs = load_split_pairs(dataset_root=dataset_root, split_name=split_name)
    if max_images is not None:
        pairs = pairs[:max_images]

    records = []
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
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Mask is not binary for {mask_rel}: values={sorted(values)}")

        fg_pixels = int(mask.sum())
        total_pixels = int(mask.size)
        records.append(
            SampleRecord(
                image_id=image_path.stem,
                image_rel=image_rel,
                mask_rel=mask_rel,
                image_path=str(image_path),
                mask_path=str(mask_path),
                width=width,
                height=height,
                fg_pixels=fg_pixels,
                total_pixels=total_pixels,
                fg_ratio=float(fg_pixels / total_pixels),
                empty_gt=fg_pixels == 0,
            )
        )
    return records


def compute_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    effective_patch = min(length, patch_size)
    starts = list(range(0, max(length - effective_patch, 0) + 1, stride))
    if not starts:
        starts = [0]
    last_start = length - effective_patch
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def build_patch_plan(width: int, height: int, patch_size: int, stride: int, image_id: str) -> PatchPlan:
    x_starts = compute_patch_starts(length=width, patch_size=patch_size, stride=stride)
    y_starts = compute_patch_starts(length=height, patch_size=patch_size, stride=stride)

    patches = []
    coverage = np.zeros((height, width), dtype=np.uint16)
    for y1 in y_starts:
        for x1 in x_starts:
            x2 = min(x1 + patch_size, width)
            y2 = min(y1 + patch_size, height)
            patches.append((x1, y1, x2, y2))
            coverage[y1:y2, x1:x2] += 1

    return PatchPlan(
        image_id=image_id,
        width=width,
        height=height,
        patch_size=patch_size,
        stride=stride,
        num_patches=len(patches),
        coverage_ok=bool(coverage.min() >= 1),
        min_coverage=int(coverage.min()),
        max_coverage=int(coverage.max()),
        patches=tuple(patches),
    )


def build_patch_plan_rows(samples: list[SampleRecord], patch_size: int, stride: int):
    plans = {}
    rows = []
    for sample in samples:
        plan = build_patch_plan(
            width=sample.width,
            height=sample.height,
            patch_size=patch_size,
            stride=stride,
            image_id=sample.image_id,
        )
        plans[sample.image_id] = plan
        rows.append(
            {
                "image_id": sample.image_id,
                "width": sample.width,
                "height": sample.height,
                "patch_size": patch_size,
                "stride": stride,
                "num_patches": plan.num_patches,
                "coverage_ok": str(plan.coverage_ok).lower(),
                "min_coverage": plan.min_coverage,
                "max_coverage": plan.max_coverage,
            }
        )
    return plans, rows


def load_rgb_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_binary_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("1"), dtype=np.uint8)


def build_prediction_mask(probability_map: np.ndarray, stage_spec: MmsegStageSpec) -> np.ndarray:
    return (probability_map > stage_spec.threshold).astype(np.uint8)


def confusion_counts(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[int, int, int, int]:
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
    fn = int(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())
    tn = int(np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum())
    return tp, fp, fn, tn


def save_binary_mask(path: Path, binary_mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((binary_mask.astype(np.uint8) * 255)).save(path)


def save_probability_map(path: Path, probability_map: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = np.clip(np.round(probability_map * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(quantized).save(path)


def build_overlay(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    tp = np.logical_and(pred_mask == 1, gt_mask == 1)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)
    alpha = 0.55
    for mask, color in (
        (tp, np.array([0, 255, 0], dtype=np.float32)),
        (fp, np.array([255, 0, 0], dtype=np.float32)),
        (fn, np.array([255, 255, 0], dtype=np.float32)),
    ):
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_overlay(path: Path, image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(build_overlay(image_rgb=image_rgb, gt_mask=gt_mask, pred_mask=pred_mask)).save(path)


def mean_and_median(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanmedian(arr))


def max_or_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return math.nan
    return float(max(valid))


def format_profile_value(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def safe_metric(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return float(numerator / denominator)


def preprocess_patch(patch_rgb: np.ndarray, model_input_size: int) -> np.ndarray:
    resized = cv2.resize(
        patch_rgb,
        dsize=(model_input_size, model_input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD,
        dtype=np.float32,
    )
    return np.transpose(normalized, (2, 0, 1))


def build_mmseg_meta(model_input_size: int):
    return {
        "ori_shape": (model_input_size, model_input_size, 3),
        "img_shape": (model_input_size, model_input_size, 3),
        "pad_shape": (model_input_size, model_input_size, 3),
        "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "flip": False,
        "flip_direction": None,
    }


def predict_full_probability_map(
    model,
    image_rgb: np.ndarray,
    patch_plan,
    model_input_size: int,
    patch_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    prob_sum = np.zeros((height, width), dtype=np.float32)
    prob_count = np.zeros((height, width), dtype=np.float32)
    patches = list(patch_plan.patches)
    segmentor = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for start_index in range(0, len(patches), patch_batch_size):
            batch_patches = patches[start_index:start_index + patch_batch_size]
            batch_input = np.stack(
                [
                    preprocess_patch(
                        patch_rgb=image_rgb[y1:y2, x1:x2],
                        model_input_size=model_input_size,
                    )
                    for x1, y1, x2, y2 in batch_patches
                ],
                axis=0,
            )
            batch_tensor = torch.as_tensor(batch_input, device=device, dtype=torch.float32)
            img_metas = [copy.deepcopy(build_mmseg_meta(model_input_size)) for _ in batch_patches]
            probabilities = segmentor.inference(batch_tensor, img_metas, rescale=False)
            crack_probabilities = probabilities[:, 1, :, :].detach().cpu().numpy().astype(np.float32)

            for probability_patch, (x1, y1, x2, y2) in zip(crack_probabilities, batch_patches):
                patch_width = x2 - x1
                patch_height = y2 - y1
                resized_probability = cv2.resize(
                    probability_patch,
                    dsize=(patch_width, patch_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                prob_sum[y1:y2, x1:x2] += resized_probability
                prob_count[y1:y2, x1:x2] += 1.0

    if np.any(prob_count <= 0):
        raise RuntimeError("Patch stitching left uncovered pixels; check patch-size/stride logic.")
    return prob_sum / prob_count


def evaluate_stage(
    stage_spec: MmsegStageSpec,
    model,
    samples,
    patch_plans,
    args,
    output_dir: Path,
    device: torch.device,
    model_load_sec: float,
):
    per_image_rows = []
    saved_count = 0

    for sample_index, sample in enumerate(samples, start=1):
        print(
            f"[{stage_spec.key}] {sample_index}/{len(samples)} "
            f"{sample.image_id} ({sample.width}x{sample.height})"
        )
        runtime_start = time.perf_counter()
        image_rgb = load_rgb_image(sample.image_path)
        gt_mask = load_binary_mask(sample.mask_path)
        load_sec = time.perf_counter() - runtime_start
        patch_plan = patch_plans[sample.image_id]
        reset_peak_memory_stats(device)
        synchronize_device(device)

        predict_start = time.perf_counter()
        probability_map = predict_full_probability_map(
            model=model,
            image_rgb=image_rgb,
            patch_plan=patch_plan,
            model_input_size=args.model_input_size,
            patch_batch_size=args.patch_batch_size,
            device=device,
        )
        synchronize_device(device)
        predict_sec = time.perf_counter() - predict_start

        postprocess_start = time.perf_counter()
        pred_mask = build_prediction_mask(probability_map=probability_map, stage_spec=stage_spec)
        tp, fp, fn, tn = confusion_counts(pred_mask=pred_mask, gt_mask=gt_mask)
        pred_pixels = int(pred_mask.sum())
        iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float("nan")
        f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        postprocess_sec = time.perf_counter() - postprocess_start
        runtime_total_sec = time.perf_counter() - runtime_start
        peak_gpu_allocated_mb, peak_gpu_reserved_mb = get_gpu_peak_memory_mb(device)
        process_rss_mb, process_peak_rss_mb = get_process_memory_mb()
        num_patch_batches = math.ceil(patch_plan.num_patches / args.patch_batch_size)
        image_megapixels = sample.total_pixels / 1_000_000.0
        patch_latency_ms = safe_metric(predict_sec * 1000.0, patch_plan.num_patches)
        patches_per_sec = safe_metric(patch_plan.num_patches, predict_sec)
        megapixels_per_sec = safe_metric(image_megapixels, predict_sec)

        per_image_rows.append(
            {
                "stage_key": stage_spec.key,
                "stage_label": stage_spec.label,
                "checkpoint_path": str(Path(stage_spec.checkpoint_path).resolve()),
                "threshold": stage_spec.threshold,
                "image_id": sample.image_id,
                "image_path": sample.image_rel,
                "mask_path": sample.mask_rel,
                "width": sample.width,
                "height": sample.height,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "iou": iou,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "fg_pixels": sample.fg_pixels,
                "pred_pixels": pred_pixels,
                "fg_ratio": sample.fg_ratio,
                "pred_ratio": float(pred_pixels / sample.total_pixels),
                "empty_gt": str(sample.empty_gt).lower(),
                "num_patches": patch_plan.num_patches,
                "num_patch_batches": num_patch_batches,
                "coverage_ok": str(patch_plan.coverage_ok).lower(),
                "image_megapixels": image_megapixels,
                "load_sec": load_sec,
                "predict_sec": predict_sec,
                "postprocess_sec": postprocess_sec,
                "runtime_total_sec": runtime_total_sec,
                "patch_latency_ms": patch_latency_ms,
                "patches_per_sec": patches_per_sec,
                "megapixels_per_sec": megapixels_per_sec,
                "peak_gpu_allocated_mb": peak_gpu_allocated_mb,
                "peak_gpu_reserved_mb": peak_gpu_reserved_mb,
                "process_rss_mb": process_rss_mb,
                "process_peak_rss_mb": process_peak_rss_mb,
            }
        )

        should_save = args.save_limit <= 0 or saved_count < args.save_limit
        if should_save:
            if not args.skip_predictions:
                save_binary_mask(
                    output_dir / "predictions" / stage_spec.key / f"{sample.image_id}_pred.png",
                    pred_mask,
                )
            if not args.skip_overlays:
                save_overlay(
                    output_dir / "overlays" / stage_spec.key / f"{sample.image_id}_overlay.png",
                    image_rgb=image_rgb,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                )
            if args.save_probability_maps:
                save_probability_map(
                    output_dir / "probability_maps_optional" / stage_spec.key / f"{sample.image_id}_prob_uint8.png",
                    probability_map=probability_map,
                )
            saved_count += 1

    if stage_spec.model_name and device.type == "cuda":
        torch.cuda.empty_cache()

    tp_total = int(np.sum([int(row["tp"]) for row in per_image_rows], dtype=np.int64))
    fp_total = int(np.sum([int(row["fp"]) for row in per_image_rows], dtype=np.int64))
    fn_total = int(np.sum([int(row["fn"]) for row in per_image_rows], dtype=np.int64))
    tn_total = int(np.sum([int(row["tn"]) for row in per_image_rows], dtype=np.int64))
    denom_iou = tp_total + fp_total + fn_total
    denom_f1 = 2 * tp_total + fp_total + fn_total
    denom_precision = tp_total + fp_total
    denom_recall = tp_total + fn_total
    total_iou = float(tp_total / denom_iou) if denom_iou > 0 else float("nan")
    total_f1 = float((2 * tp_total) / denom_f1) if denom_f1 > 0 else float("nan")
    total_precision = float(tp_total / denom_precision) if denom_precision > 0 else float("nan")
    total_recall = float(tp_total / denom_recall) if denom_recall > 0 else float("nan")

    image_ious = [float(row["iou"]) for row in per_image_rows]
    image_f1s = [float(row["f1"]) for row in per_image_rows]
    nonempty_image_ious = [float(row["iou"]) for row in per_image_rows if row["empty_gt"] == "false"]
    nonempty_image_f1s = [float(row["f1"]) for row in per_image_rows if row["empty_gt"] == "false"]
    mean_image_iou, median_image_iou = mean_and_median(image_ious)
    mean_image_f1, median_image_f1 = mean_and_median(image_f1s)
    mean_image_iou_nonempty, median_image_iou_nonempty = mean_and_median(nonempty_image_ious)
    mean_image_f1_nonempty, median_image_f1_nonempty = mean_and_median(nonempty_image_f1s)
    image_runtime_secs = [float(row["runtime_total_sec"]) for row in per_image_rows]
    image_predict_secs = [float(row["predict_sec"]) for row in per_image_rows]
    patch_latency_values = [float(row["patch_latency_ms"]) for row in per_image_rows]
    patches_per_sec_values = [float(row["patches_per_sec"]) for row in per_image_rows]
    megapixels_per_sec_values = [float(row["megapixels_per_sec"]) for row in per_image_rows]
    peak_gpu_allocated_values = [float(row["peak_gpu_allocated_mb"]) for row in per_image_rows]
    peak_gpu_reserved_values = [float(row["peak_gpu_reserved_mb"]) for row in per_image_rows]
    process_rss_values = [float(row["process_rss_mb"]) for row in per_image_rows]
    process_peak_rss_values = [float(row["process_peak_rss_mb"]) for row in per_image_rows]
    mean_image_runtime_sec, median_image_runtime_sec = mean_and_median(image_runtime_secs)
    mean_image_predict_sec, median_image_predict_sec = mean_and_median(image_predict_secs)
    mean_patch_latency_ms, median_patch_latency_ms = mean_and_median(patch_latency_values)
    mean_patches_per_sec, median_patches_per_sec = mean_and_median(patches_per_sec_values)
    mean_megapixels_per_sec, median_megapixels_per_sec = mean_and_median(megapixels_per_sec_values)
    runtime_total_sec = float(np.sum(image_runtime_secs, dtype=np.float64))
    predict_total_sec = float(np.sum(image_predict_secs, dtype=np.float64))
    total_patches = int(np.sum([int(row["num_patches"]) for row in per_image_rows], dtype=np.int64))
    total_megapixels = float(np.sum([float(row["image_megapixels"]) for row in per_image_rows], dtype=np.float64))

    aggregate_row = {
        "stage_key": stage_spec.key,
        "stage_label": stage_spec.label,
        "checkpoint_path": str(Path(stage_spec.checkpoint_path).resolve()),
        "threshold": stage_spec.threshold,
        "num_images": len(samples),
        "num_nonempty_gt": sum(not sample.empty_gt for sample in samples),
        "num_empty_gt": sum(sample.empty_gt for sample in samples),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "tn": tn_total,
        "iou": total_iou,
        "f1": total_f1,
        "precision": total_precision,
        "recall": total_recall,
        "mean_image_iou": mean_image_iou,
        "median_image_iou": median_image_iou,
        "mean_image_f1": mean_image_f1,
        "median_image_f1": median_image_f1,
        "mean_image_iou_nonempty_gt": mean_image_iou_nonempty,
        "median_image_iou_nonempty_gt": median_image_iou_nonempty,
        "mean_image_f1_nonempty_gt": mean_image_f1_nonempty,
        "median_image_f1_nonempty_gt": median_image_f1_nonempty,
        "model_load_sec": model_load_sec,
        "runtime_total_sec": runtime_total_sec,
        "predict_total_sec": predict_total_sec,
        "mean_image_runtime_sec": mean_image_runtime_sec,
        "median_image_runtime_sec": median_image_runtime_sec,
        "mean_image_predict_sec": mean_image_predict_sec,
        "median_image_predict_sec": median_image_predict_sec,
        "mean_patch_latency_ms": mean_patch_latency_ms,
        "median_patch_latency_ms": median_patch_latency_ms,
        "mean_patches_per_sec": mean_patches_per_sec,
        "median_patches_per_sec": median_patches_per_sec,
        "global_patches_per_sec": safe_metric(total_patches, predict_total_sec),
        "mean_megapixels_per_sec": mean_megapixels_per_sec,
        "median_megapixels_per_sec": median_megapixels_per_sec,
        "global_megapixels_per_sec": safe_metric(total_megapixels, predict_total_sec),
        "global_runtime_megapixels_per_sec": safe_metric(total_megapixels, runtime_total_sec),
        "total_patches": total_patches,
        "total_megapixels": total_megapixels,
        "peak_gpu_allocated_mb": max_or_nan(peak_gpu_allocated_values),
        "peak_gpu_reserved_mb": max_or_nan(peak_gpu_reserved_values),
        "peak_process_rss_mb": max_or_nan(process_rss_values),
        "peak_process_hwm_mb": max_or_nan(process_peak_rss_values),
    }
    return aggregate_row, per_image_rows


def build_eval_config(args, samples, stage_spec: MmsegStageSpec):
    unique_shapes = sorted({f"{sample.width}x{sample.height}" for sample in samples})
    single_resolution = unique_shapes[0] if len(unique_shapes) == 1 else None
    return {
        "dataset": "SAM799-CVAT",
        "dataset_role": "fixed external test only",
        "split_name": args.split,
        "evaluation_scope": "subset_smoke_test" if args.max_images is not None else "full_external_test",
        "num_images": len(samples),
        "num_positive": sum(not sample.empty_gt for sample in samples),
        "num_empty_negative": sum(sample.empty_gt for sample in samples),
        "image_resolution": single_resolution,
        "image_resolutions": unique_shapes,
        "used_for_training": False,
        "used_for_validation": False,
        "used_for_threshold_selection": False,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "model_input_size": args.model_input_size,
        "patch_batch_size": args.patch_batch_size,
        "stitching": "probability_average",
        "threshold_policy": "predefined from primary Kaggle validation protocol",
        "profiling": {
            "enabled": True,
            "time_unit": "seconds",
            "memory_unit": "MiB",
            "runtime_scope": (
                "Per-image runtime covers image load, patchwise inference/stitching, and "
                "threshold/postprocess/metrics; it excludes optional prediction or overlay file writes."
            ),
            "predict_scope": (
                "Predict time covers patch preprocessing, batched forward passes, resize-back, "
                "and probability averaging."
            ),
            "gpu_peak_scope": "Peak GPU memory is reset after model load and tracked per image.",
        },
        "stages": [
            {
                **asdict(stage_spec),
                "checkpoint_path": str(Path(stage_spec.checkpoint_path).resolve()),
                "config_path": str(Path(stage_spec.config_path).resolve()),
                "mmseg_root": str(Path(stage_spec.mmseg_root).resolve()),
            }
        ],
    }


def write_readme(output_dir: Path, args, stage_spec: MmsegStageSpec, samples):
    lines = [
        "# SAM799 External Patchwise Evaluation",
        "",
        "This directory contains fixed-protocol external evaluation artifacts for a "
        "single DAFormer/HRDA reviewer-baseline checkpoint on SAM799-CVAT.",
        "",
        "Protocol summary:",
        f"- Dataset root: `{Path(args.dataset_root).resolve()}`",
        f"- Split evaluated: `{args.split}`",
        f"- Images evaluated: `{len(samples)}`",
        f"- Patch size / stride: `{args.patch_size}` / `{args.stride}`",
        f"- Model input size: `{args.model_input_size}`",
        f"- Stitching: `probability_average`",
        "- Training, validation, and threshold selection on SAM799-CVAT: `false`",
        "",
        "Stage:",
        (
            f"- `{stage_spec.key}`: `{stage_spec.label}` | threshold `{stage_spec.threshold}` | "
            f"checkpoint `{Path(stage_spec.checkpoint_path).resolve()}` | "
            f"config `{Path(stage_spec.config_path).resolve()}`"
        ),
        "",
        "Generated files:",
        "- `eval_config.json`",
        "- `sam799_manifest.csv`",
        "- `per_image_patch_counts.csv`",
        "- `per_image_metrics.csv`",
        "- `aggregate_metrics.csv`",
        "- `profiling_summary.md`",
        "- `predictions/` and `overlays/` (unless skipped)",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_profiling_summary(output_dir: Path, aggregate_rows: list[dict[str, object]]):
    lines = [
        "# Profiling Summary",
        "",
        "Patchwise runtime and memory summary on SAM799-CVAT.",
        "",
        "- `runtime_total_sec`: image load + patchwise inference/stitching + threshold/postprocess/metrics; excludes optional asset writes",
        "- `predict_total_sec`: patch preprocessing + batched forward passes + resize-back + probability averaging",
        "- `global_patches_per_sec` and `global_megapixels_per_sec` are computed from aggregate predict time",
        "- GPU peaks are reported only when running on CUDA; CPU-only runs show `n/a`",
        "",
        "| Stage | Model load (s) | Mean image runtime (s) | Mean image predict (s) | Global patches/s | Global MPix/s | Peak GPU alloc (MiB) | Peak RSS (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {stage} | {model_load} | {image_runtime} | {image_predict} | {patches_per_sec} | {mpix_per_sec} | {gpu_alloc} | {rss} |".format(
                stage=row["stage_label"],
                model_load=format_profile_value(row["model_load_sec"]),
                image_runtime=format_profile_value(row["mean_image_runtime_sec"]),
                image_predict=format_profile_value(row["mean_image_predict_sec"]),
                patches_per_sec=format_profile_value(row["global_patches_per_sec"]),
                mpix_per_sec=format_profile_value(row["global_megapixels_per_sec"]),
                gpu_alloc=format_profile_value(row["peak_gpu_allocated_mb"]),
                rss=format_profile_value(row["peak_process_rss_mb"]),
            )
        )

    (output_dir / "profiling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    stage_spec = MmsegStageSpec(
        key=args.stage_key,
        label=args.stage_label,
        checkpoint_path=args.checkpoint_path,
        threshold=args.threshold,
        config_path=args.config,
        mmseg_root=args.mmseg_root,
        model_name=args.model_name,
    )

    add_mmseg_to_path(stage_spec.mmseg_root)
    mmcv, MMDataParallel, build_segmentor = import_mmseg_modules()
    cfg = mmcv.Config.fromfile(stage_spec.config_path)
    model_load_start = time.perf_counter()
    model = init_model(cfg, stage_spec.checkpoint_path, device, build_segmentor, MMDataParallel)
    synchronize_device(device)
    model_load_sec = time.perf_counter() - model_load_start

    samples = load_samples(
        dataset_root=dataset_root,
        split_name=args.split,
        max_images=args.max_images,
    )
    patch_plans, patch_plan_rows = build_patch_plan_rows(
        samples=samples,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    print(f"Using device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Split: {args.split}")
    print(f"Images: {len(samples)}")
    print(f"Stage: {stage_spec.key}")
    print(f"Patch size / stride: {args.patch_size} / {args.stride}")

    eval_config = build_eval_config(args=args, samples=samples, stage_spec=stage_spec)
    (output_dir / "eval_config.json").write_text(json.dumps(eval_config, indent=2), encoding="utf-8")
    write_readme(output_dir=output_dir, args=args, stage_spec=stage_spec, samples=samples)
    write_csv(output_dir / "per_image_patch_counts.csv", patch_plan_rows)

    aggregate_row, per_image_rows = evaluate_stage(
        stage_spec=stage_spec,
        model=model,
        samples=samples,
        patch_plans=patch_plans,
        args=args,
        output_dir=output_dir,
        device=device,
        model_load_sec=model_load_sec,
    )
    write_csv(output_dir / "aggregate_metrics.csv", [aggregate_row])
    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_profiling_summary(output_dir=output_dir, aggregate_rows=[aggregate_row])

    manifest_path = Path(args.manifest_csv).resolve()
    if manifest_path.exists():
        manifest_copy_path = output_dir / "sam799_manifest.csv"
        if manifest_copy_path.resolve() != manifest_path:
            shutil.copy2(manifest_path, manifest_copy_path)
        print(f"Manifest already present at: {manifest_path}")
    else:
        print(
            "Manifest CSV not found in the requested location. "
            "Run scripts/data/build_sam799_manifest.py to create it."
        )

    print(f"Wrote eval config to: {output_dir / 'eval_config.json'}")
    print(f"Wrote patch-count rows to: {output_dir / 'per_image_patch_counts.csv'}")
    print(f"Wrote per-image metrics to: {output_dir / 'per_image_metrics.csv'}")
    print(f"Wrote aggregate metrics to: {output_dir / 'aggregate_metrics.csv'}")
    print(f"Wrote profiling summary to: {output_dir / 'profiling_summary.md'}")


if __name__ == "__main__":
    main()
