import argparse
import csv
import gc
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crack_detection.dataset import CrackDataset
from crack_detection.model import SEGFORMER_B2_MODEL_NAME, get_model
from crack_detection.postprocess import build_postprocess_config, logits_to_binary_mask


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    checkpoint_path: str
    threshold: float
    model_name: str = "segformer-b2"
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    pretrained_model_name: str = SEGFORMER_B2_MODEL_NAME
    postprocess_min_area: int = 0
    postprocess_max_fill_ratio: float = 1.0
    postprocess_min_aspect_ratio: float = 1.0
    postprocess_max_components: int = 0


DEFAULT_STAGE_SPECS = [
    StageSpec(
        key="source_only",
        label="Source-only @ 0.5",
        checkpoint_path="checkpoints/segformer_b2_plain_360.pth",
        threshold=0.5,
    ),
    StageSpec(
        key="b1_promoted",
        label="B1 promoted TS-bank @ 0.6",
        checkpoint_path="checkpoints/segformer_b2_b1_tsbank_thr080_mean082.pth",
        threshold=0.6,
    ),
    StageSpec(
        key="b2_fs05",
        label="B2 fs05 @ 0.5",
        checkpoint_path="checkpoints/segformer_b2_b2_fs05_seed42.pth",
        threshold=0.5,
    ),
    StageSpec(
        key="b2_fs10",
        label="B2 fs10 @ 0.5",
        checkpoint_path="checkpoints/segformer_b2_b2_fs10_seed42.pth",
        threshold=0.5,
    ),
    StageSpec(
        key="b2_fs20",
        label="B2 fs20 @ 0.5",
        checkpoint_path="checkpoints/segformer_b2_b2_fs20_seed42.pth",
        threshold=0.5,
    ),
]

LEGACY_B1_SPEC = StageSpec(
    key="b1_legacy",
    label="B1 legacy raw @ 0.7",
    checkpoint_path="checkpoints/segformer_b2_b1_negbank.pth",
    threshold=0.7,
)

UPPER_BOUND_SPEC = StageSpec(
    key="upper_bound",
    label="Target upper bound @ 0.5",
    checkpoint_path="checkpoints/segformer_b2_uav_indomain_plain_360.pth",
    threshold=0.5,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate source-domain retention analysis on Crack500 test and "
            "bootstrap CIs for Kaggle test metrics without retraining."
        )
    )
    parser.add_argument("--source-dataset-root", default="CRACK500")
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--target-dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--target-split", default="test")
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--include-legacy-b1", action="store_true")
    parser.add_argument("--include-upper-bound", action="store_true")
    parser.add_argument("--stage-keys", nargs="+", default=None)
    parser.add_argument("--skip-skeleton", action="store_true")
    parser.add_argument("--save-per-sample", action="store_true")
    parser.add_argument("--output-dir", default="results/report_assets/source_retention")
    return parser.parse_args()


def build_stage_specs(args) -> list[StageSpec]:
    stage_specs = list(DEFAULT_STAGE_SPECS)
    if args.include_legacy_b1:
        stage_specs.insert(2, LEGACY_B1_SPEC)
    if args.include_upper_bound:
        stage_specs.append(UPPER_BOUND_SPEC)
    if args.stage_keys:
        wanted = set(args.stage_keys)
        stage_specs = [spec for spec in stage_specs if spec.key in wanted]
        missing = wanted.difference({spec.key for spec in stage_specs})
        if missing:
            raise ValueError(f"Unknown --stage-keys requested: {sorted(missing)}")
    if not any(spec.key == "source_only" for spec in stage_specs):
        raise ValueError("`source_only` must be included so retention ratios have a reference row.")
    return stage_specs


def mask_confusion_counts(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[int, int, int]:
    pred_flat = pred_mask.astype(np.uint8).reshape(-1)
    gt_flat = gt_mask.astype(np.uint8).reshape(-1)
    tp = int(np.logical_and(pred_flat == 1, gt_flat == 1).sum())
    fp = int(np.logical_and(pred_flat == 1, gt_flat == 0).sum())
    fn = int(np.logical_and(pred_flat == 0, gt_flat == 1).sum())
    return tp, fp, fn


def metrics_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def skeletonize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    work = (binary_mask > 0).astype(np.uint8)
    if work.sum() == 0:
        return work

    # Morphological skeletonization keeps the implementation self-contained
    # without requiring scikit-image or OpenCV contrib.
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(work, opened)
        eroded = cv2.erode(work, element)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break

    return (skeleton > 0).astype(np.uint8)


def dataset_tag(dataset_root: str, split: str) -> str:
    root_name = Path(dataset_root).name.lower()
    return f"{root_name}_{split}".replace("-", "_")


def load_model_for_stage(stage_spec: StageSpec, device: torch.device) -> torch.nn.Module:
    model = get_model(
        model_name=stage_spec.model_name,
        encoder_name=stage_spec.encoder_name,
        encoder_weights=stage_spec.encoder_weights,
        in_channels=3,
        classes=1,
        pretrained_model_name=stage_spec.pretrained_model_name,
    )
    checkpoint_path = Path(stage_spec.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def evaluate_dataset(
    model: torch.nn.Module,
    stage_spec: StageSpec,
    dataset_root: str,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    include_skeleton: bool,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    dataset = CrackDataset(root=dataset_root, split=split, img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    postprocess_config = build_postprocess_config(
        min_area=stage_spec.postprocess_min_area,
        max_fill_ratio=stage_spec.postprocess_max_fill_ratio,
        min_aspect_ratio=stage_spec.postprocess_min_aspect_ratio,
        max_components=stage_spec.postprocess_max_components,
    )

    per_sample_rows: list[dict[str, float | int | str]] = []
    sample_index = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            pred_masks = logits_to_binary_mask(
                logits=logits,
                threshold=stage_spec.threshold,
                postprocess_config=postprocess_config,
            )
            pred_masks_np = pred_masks.squeeze(1).cpu().numpy().astype(np.uint8)
            gt_masks_np = masks.cpu().numpy().astype(np.uint8)

            for pred_mask, gt_mask in zip(pred_masks_np, gt_masks_np):
                image_path, mask_path = dataset.samples[sample_index]
                tp, fp, fn = mask_confusion_counts(pred_mask, gt_mask)
                metrics = metrics_from_counts(tp, fp, fn)
                row: dict[str, float | int | str] = {
                    "stage_key": stage_spec.key,
                    "stage_label": stage_spec.label,
                    "dataset_root": dataset_root,
                    "split": split,
                    "sample_index": sample_index,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "threshold": stage_spec.threshold,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "iou": float(metrics["iou"]),
                    "f1": float(metrics["f1"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                }

                if include_skeleton:
                    pred_skeleton = skeletonize_binary_mask(pred_mask)
                    gt_skeleton = skeletonize_binary_mask(gt_mask)
                    sk_tp, sk_fp, sk_fn = mask_confusion_counts(pred_skeleton, gt_skeleton)
                    sk_metrics = metrics_from_counts(sk_tp, sk_fp, sk_fn)
                    row.update(
                        {
                            "skeleton_tp": sk_tp,
                            "skeleton_fp": sk_fp,
                            "skeleton_fn": sk_fn,
                            "skeleton_iou": float(sk_metrics["iou"]),
                            "skeleton_f1": float(sk_metrics["f1"]),
                        }
                    )

                per_sample_rows.append(row)
                sample_index += 1

    tp_total = np.sum([int(row["tp"]) for row in per_sample_rows], dtype=np.int64)
    fp_total = np.sum([int(row["fp"]) for row in per_sample_rows], dtype=np.int64)
    fn_total = np.sum([int(row["fn"]) for row in per_sample_rows], dtype=np.int64)
    total_metrics = metrics_from_counts(tp_total, fp_total, fn_total)

    summary: dict[str, float | int | str] = {
        "stage_key": stage_spec.key,
        "stage_label": stage_spec.label,
        "dataset_root": dataset_root,
        "split": split,
        "num_samples": len(dataset),
        "checkpoint_path": str(Path(stage_spec.checkpoint_path).resolve()),
        "threshold": stage_spec.threshold,
        "iou": float(total_metrics["iou"]),
        "f1": float(total_metrics["f1"]),
        "precision": float(total_metrics["precision"]),
        "recall": float(total_metrics["recall"]),
    }

    if include_skeleton:
        sk_tp_total = np.sum([int(row["skeleton_tp"]) for row in per_sample_rows], dtype=np.int64)
        sk_fp_total = np.sum([int(row["skeleton_fp"]) for row in per_sample_rows], dtype=np.int64)
        sk_fn_total = np.sum([int(row["skeleton_fn"]) for row in per_sample_rows], dtype=np.int64)
        sk_total_metrics = metrics_from_counts(sk_tp_total, sk_fp_total, sk_fn_total)
        summary.update(
            {
                "skeleton_iou": float(sk_total_metrics["iou"]),
                "skeleton_f1": float(sk_total_metrics["f1"]),
            }
        )

    return summary, per_sample_rows


def bootstrap_confidence_intervals(
    per_sample_rows: list[dict[str, float | int | str]],
    num_bootstrap_samples: int,
    seed: int,
    include_skeleton: bool,
) -> dict[str, float]:
    if not per_sample_rows:
        raise ValueError("Cannot bootstrap an empty evaluation set.")

    rng = np.random.default_rng(seed)
    sample_count = len(per_sample_rows)
    indices = rng.integers(0, sample_count, size=(num_bootstrap_samples, sample_count))

    tp = np.array([int(row["tp"]) for row in per_sample_rows], dtype=np.int64)
    fp = np.array([int(row["fp"]) for row in per_sample_rows], dtype=np.int64)
    fn = np.array([int(row["fn"]) for row in per_sample_rows], dtype=np.int64)

    tp_sum = tp[indices].sum(axis=1, dtype=np.int64)
    fp_sum = fp[indices].sum(axis=1, dtype=np.int64)
    fn_sum = fn[indices].sum(axis=1, dtype=np.int64)
    metric_arrays = metrics_from_counts(tp_sum, fp_sum, fn_sum)

    ci = {
        "bootstrap_samples": num_bootstrap_samples,
    }
    for metric_name, values in metric_arrays.items():
        ci[f"{metric_name}_ci_low"] = float(np.quantile(values, 0.025))
        ci[f"{metric_name}_ci_high"] = float(np.quantile(values, 0.975))

    if include_skeleton:
        sk_tp = np.array([int(row["skeleton_tp"]) for row in per_sample_rows], dtype=np.int64)
        sk_fp = np.array([int(row["skeleton_fp"]) for row in per_sample_rows], dtype=np.int64)
        sk_fn = np.array([int(row["skeleton_fn"]) for row in per_sample_rows], dtype=np.int64)

        sk_tp_sum = sk_tp[indices].sum(axis=1, dtype=np.int64)
        sk_fp_sum = sk_fp[indices].sum(axis=1, dtype=np.int64)
        sk_fn_sum = sk_fn[indices].sum(axis=1, dtype=np.int64)
        sk_metric_arrays = metrics_from_counts(sk_tp_sum, sk_fp_sum, sk_fn_sum)
        for metric_name in ("iou", "f1"):
            values = sk_metric_arrays[metric_name]
            ci[f"skeleton_{metric_name}_ci_low"] = float(np.quantile(values, 0.025))
            ci[f"skeleton_{metric_name}_ci_high"] = float(np.quantile(values, 0.975))

    return ci


def write_csv(csv_path: Path, rows: list[dict[str, float | int | str]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {csv_path}")

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def build_combined_rows(
    stage_specs: list[StageSpec],
    source_results: dict[str, dict[str, float | int | str]],
    target_results: dict[str, dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    baseline_source_iou = float(source_results["source_only"]["iou"])
    baseline_source_f1 = float(source_results["source_only"]["f1"])
    baseline_target_iou = float(target_results["source_only"]["iou"])
    baseline_target_f1 = float(target_results["source_only"]["f1"])

    combined_rows: list[dict[str, float | int | str]] = []
    for spec in stage_specs:
        source_row = source_results[spec.key]
        target_row = target_results[spec.key]
        row: dict[str, float | int | str] = {
            "stage_key": spec.key,
            "stage_label": spec.label,
            "checkpoint_path": source_row["checkpoint_path"],
            "threshold": spec.threshold,
            "crack500_iou": source_row["iou"],
            "crack500_f1": source_row["f1"],
            "crack500_precision": source_row["precision"],
            "crack500_recall": source_row["recall"],
            "kaggle_iou": target_row["iou"],
            "kaggle_f1": target_row["f1"],
            "kaggle_precision": target_row["precision"],
            "kaggle_recall": target_row["recall"],
            "crack500_iou_retention_ratio": float(source_row["iou"]) / baseline_source_iou,
            "crack500_f1_retention_ratio": float(source_row["f1"]) / baseline_source_f1,
            "crack500_iou_delta_vs_source": float(source_row["iou"]) - baseline_source_iou,
            "crack500_f1_delta_vs_source": float(source_row["f1"]) - baseline_source_f1,
            "kaggle_iou_gain_vs_source": float(target_row["iou"]) - baseline_target_iou,
            "kaggle_f1_gain_vs_source": float(target_row["f1"]) - baseline_target_f1,
        }

        for metric_name in ("iou", "f1", "precision", "recall"):
            low_key = f"{metric_name}_ci_low"
            high_key = f"{metric_name}_ci_high"
            if low_key in target_row and high_key in target_row:
                row[f"kaggle_{metric_name}_ci_low"] = target_row[low_key]
                row[f"kaggle_{metric_name}_ci_high"] = target_row[high_key]

        if "skeleton_iou" in source_row:
            row["crack500_skeleton_iou"] = source_row["skeleton_iou"]
            row["crack500_skeleton_f1"] = source_row["skeleton_f1"]
        if "skeleton_iou" in target_row:
            row["kaggle_skeleton_iou"] = target_row["skeleton_iou"]
            row["kaggle_skeleton_f1"] = target_row["skeleton_f1"]
            if "skeleton_iou_ci_low" in target_row:
                row["kaggle_skeleton_iou_ci_low"] = target_row["skeleton_iou_ci_low"]
                row["kaggle_skeleton_iou_ci_high"] = target_row["skeleton_iou_ci_high"]
                row["kaggle_skeleton_f1_ci_low"] = target_row["skeleton_f1_ci_low"]
                row["kaggle_skeleton_f1_ci_high"] = target_row["skeleton_f1_ci_high"]

        combined_rows.append(row)

    return combined_rows


def build_summary_markdown(
    combined_rows: list[dict[str, float | int | str]],
    source_tag: str,
    target_tag: str,
    include_skeleton: bool,
) -> str:
    stage_order = [row["stage_key"] for row in combined_rows]
    combined_by_key = {row["stage_key"]: row for row in combined_rows}
    source_row = combined_by_key["source_only"]

    lines = [
        "# Source-Domain Retention Analysis",
        "",
        "This report back-tests each frozen target-stage checkpoint on `Crack500 test` "
        "using the same operating point used for the target-side row.",
        "",
        f"- Source reference (`{source_tag}`): "
        f"`IoU {format_metric(float(source_row['crack500_iou']))}`, "
        f"`F1 {format_metric(float(source_row['crack500_f1']))}`",
        f"- Target reference (`{target_tag}`): "
        f"`IoU {format_metric(float(source_row['kaggle_iou']))}`, "
        f"`F1 {format_metric(float(source_row['kaggle_f1']))}`",
        "",
        "## Main Findings",
        "",
    ]

    for stage_key in stage_order:
        if stage_key == "source_only":
            continue
        row = combined_by_key[stage_key]
        source_retention = float(row["crack500_iou_retention_ratio"]) * 100.0
        lines.append(
            "- "
            f"`{row['stage_label']}` keeps `Crack500 IoU {format_metric(float(row['crack500_iou']))}` "
            f"({source_retention:.1f}% of source-only; "
            f"delta `{float(row['crack500_iou_delta_vs_source']):+.4f}`) while raising "
            f"`Kaggle IoU` to `{format_metric(float(row['kaggle_iou']))}` "
            f"(gain `{float(row['kaggle_iou_gain_vs_source']):+.4f}` vs source-only; "
            f"95% bootstrap CI "
            f"`[{format_metric(float(row['kaggle_iou_ci_low']))}, {format_metric(float(row['kaggle_iou_ci_high']))}]`)."
        )

    lines.extend(
        [
            "",
            "## Combined Table",
            "",
            "| Stage | Crack500 IoU | Retention | Kaggle IoU | Kaggle 95% CI | Kaggle F1 | Kaggle F1 95% CI |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )

    for row in combined_rows:
        lines.append(
            "| "
            f"{row['stage_label']} | "
            f"{format_metric(float(row['crack500_iou']))} | "
            f"{float(row['crack500_iou_retention_ratio']) * 100.0:.1f}% | "
            f"{format_metric(float(row['kaggle_iou']))} | "
            f"[{format_metric(float(row['kaggle_iou_ci_low']))}, {format_metric(float(row['kaggle_iou_ci_high']))}] | "
            f"{format_metric(float(row['kaggle_f1']))} | "
            f"[{format_metric(float(row['kaggle_f1_ci_low']))}, {format_metric(float(row['kaggle_f1_ci_high']))}] |"
        )

    if include_skeleton:
        lines.extend(
            [
                "",
                "## Skeleton Metric",
                "",
                "Skeleton scores are computed after a lightweight morphological skeletonization of "
                "both prediction and ground-truth masks. They are stricter than region IoU/F1 and "
                "mainly reflect centerline retention.",
                "",
                "| Stage | Crack500 skeleton F1 | Kaggle skeleton F1 | Kaggle skeleton F1 95% CI |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for row in combined_rows:
            lines.append(
                "| "
                f"{row['stage_label']} | "
                f"{format_metric(float(row['crack500_skeleton_f1']))} | "
                f"{format_metric(float(row['kaggle_skeleton_f1']))} | "
                f"[{format_metric(float(row['kaggle_skeleton_f1_ci_low']))}, "
                f"{format_metric(float(row['kaggle_skeleton_f1_ci_high']))}] |"
            )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- `Crack500 test` is used as the source-domain retention back-test.",
            "- `UAV_Crack_Segmentation_Kaggle test` is resampled at the image level with replacement.",
            "- Each bootstrap replicate recomputes dataset-level metrics from summed per-image `tp/fp/fn`.",
            "- Point estimates in this report are recomputed from the same per-image counts for bootstrap consistency, so they can differ slightly from older batch-averaged rows in `results/experiments.csv`.",
            "- No checkpoint is retrained for this report.",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_specs = build_stage_specs(args)
    include_skeleton = not args.skip_skeleton
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_results: dict[str, dict[str, float | int | str]] = {}
    target_results: dict[str, dict[str, float | int | str]] = {}

    source_tag = dataset_tag(args.source_dataset_root, args.source_split)
    target_tag = dataset_tag(args.target_dataset_root, args.target_split)
    per_sample_dir = output_dir / "per_sample"

    print(f"Using device: {device}")
    print(f"Stages: {[spec.key for spec in stage_specs]}")
    print(f"Source back-test: {args.source_dataset_root} / {args.source_split}")
    print(f"Target bootstrap set: {args.target_dataset_root} / {args.target_split}")

    for stage_spec in stage_specs:
        print(f"\n=== Evaluating {stage_spec.label} ===")
        model = load_model_for_stage(stage_spec, device)

        source_summary, source_per_sample = evaluate_dataset(
            model=model,
            stage_spec=stage_spec,
            dataset_root=args.source_dataset_root,
            split=args.source_split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            include_skeleton=include_skeleton,
        )
        source_results[stage_spec.key] = source_summary

        target_summary, target_per_sample = evaluate_dataset(
            model=model,
            stage_spec=stage_spec,
            dataset_root=args.target_dataset_root,
            split=args.target_split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            include_skeleton=include_skeleton,
        )
        target_summary.update(
            bootstrap_confidence_intervals(
                per_sample_rows=target_per_sample,
                num_bootstrap_samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
                include_skeleton=include_skeleton,
            )
        )
        target_results[stage_spec.key] = target_summary

        if args.save_per_sample:
            write_csv(
                per_sample_dir / f"{source_tag}__{stage_spec.key}.csv",
                source_per_sample,
            )
            write_csv(
                per_sample_dir / f"{target_tag}__{stage_spec.key}.csv",
                target_per_sample,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    source_rows = [source_results[spec.key] for spec in stage_specs]
    target_rows = [target_results[spec.key] for spec in stage_specs]
    combined_rows = build_combined_rows(stage_specs, source_results, target_results)

    write_csv(output_dir / f"{source_tag}_retention.csv", source_rows)
    write_csv(output_dir / f"{target_tag}_bootstrap_ci.csv", target_rows)
    write_csv(output_dir / "combined_summary.csv", combined_rows)

    summary_md = build_summary_markdown(
        combined_rows=combined_rows,
        source_tag=source_tag,
        target_tag=target_tag,
        include_skeleton=include_skeleton,
    )
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"\nSaved source retention table to {output_dir / f'{source_tag}_retention.csv'}")
    print(f"Saved target bootstrap CI table to {output_dir / f'{target_tag}_bootstrap_ci.csv'}")
    print(f"Saved combined summary to {output_dir / 'combined_summary.csv'}")
    print(f"Saved narrative summary to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
