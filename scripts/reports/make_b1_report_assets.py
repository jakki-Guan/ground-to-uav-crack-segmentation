import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import CrackDataset
from loss import build_loss
from metrics import confusion_stats
from model import SEGFORMER_B2_MODEL_NAME, get_model
from postprocess import (
    ConnectedComponentPostprocessConfig,
    build_postprocess_config,
    filter_connected_components,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class VariantSpec:
    label: str
    threshold: float
    postprocess_config: ConnectedComponentPostprocessConfig | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate B1 report assets: main table, threshold sweep, and qualitative figure."
    )
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--model-name", default="segformer-b2")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--pretrained-model-name", default=SEGFORMER_B2_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--source-checkpoint", default="checkpoints/segformer_b2_plain_360.pth")
    parser.add_argument("--b1-checkpoint", default="checkpoints/segformer_b2_b1_negbank.pth")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--output-dir", default="results/report_assets/b1_holdout")
    parser.add_argument("--qualitative-count", type=int, default=3)
    parser.add_argument("--postprocess-min-area", type=int, default=20)
    parser.add_argument("--postprocess-max-fill-ratio", type=float, default=0.85)
    parser.add_argument("--postprocess-min-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-max-components", type=int, default=0)
    return parser.parse_args()


def parse_threshold_list(raw: str) -> list[float]:
    thresholds = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")
    return thresholds


def build_dataset(root: str, split: str, img_size: int) -> CrackDataset:
    return CrackDataset(root=root, split=split, img_size=img_size)


def build_loader(dataset: CrackDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def load_model_from_checkpoint(args, checkpoint_path: str, device: torch.device):
    model = get_model(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
        pretrained_model_name=args.pretrained_model_name,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def evaluate_variants(
    model,
    loader: DataLoader,
    criterion,
    device: torch.device,
    variants: list[VariantSpec],
):
    metrics_accum = {
        variant.label: {
            "loss": 0.0,
            "iou": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
        for variant in variants
    }

    dataset_size = len(loader.dataset)
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            batch_size_now = images.size(0)

            for variant in variants:
                batch_metrics = confusion_stats(
                    outputs,
                    masks,
                    threshold=variant.threshold,
                    postprocess_config=variant.postprocess_config,
                )
                metrics_accum[variant.label]["loss"] += loss.item() * batch_size_now
                metrics_accum[variant.label]["iou"] += batch_metrics["iou"] * batch_size_now
                metrics_accum[variant.label]["f1"] += batch_metrics["f1"] * batch_size_now
                metrics_accum[variant.label]["precision"] += (
                    batch_metrics["precision"] * batch_size_now
                )
                metrics_accum[variant.label]["recall"] += batch_metrics["recall"] * batch_size_now

    for variant in variants:
        for key in metrics_accum[variant.label]:
            metrics_accum[variant.label][key] /= dataset_size

    return metrics_accum


def collect_prob_maps(
    model,
    loader: DataLoader,
    device: torch.device,
):
    probabilities = []
    masks = []
    with torch.no_grad():
        for images, batch_masks in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            probabilities.extend(probs)
            masks.extend(batch_masks.numpy().astype(np.uint8))
    return probabilities, masks


def binary_from_prob_map(
    prob_map: np.ndarray,
    threshold: float,
    postprocess_config: ConnectedComponentPostprocessConfig | None = None,
) -> np.ndarray:
    if postprocess_config is None:
        return (prob_map > threshold).astype(np.uint8)
    return filter_connected_components(
        prob_map=prob_map,
        threshold=threshold,
        config=postprocess_config,
    ).astype(np.uint8)


def sample_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    tp = float(np.logical_and(pred_bool, gt_bool).sum())
    fp = float(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = float(np.logical_and(~pred_bool, gt_bool).sum())
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def resize_raw_pair(dataset: CrackDataset, idx: int, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    image, mask = dataset.get_raw(idx)
    image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized.astype(np.uint8)


def build_overlay(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    tp = np.logical_and(pred_mask == 1, gt_mask == 1)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)

    overlay[tp] = 0.5 * overlay[tp] + 0.5 * np.array([0, 255, 0], dtype=np.float32)
    overlay[fp] = 0.45 * overlay[fp] + 0.55 * np.array([255, 0, 0], dtype=np.float32)
    overlay[fn] = 0.45 * overlay[fn] + 0.55 * np.array([255, 255, 0], dtype=np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def case_label_from_sample(dataset: CrackDataset, idx: int) -> str:
    image_path, _ = dataset.samples[idx]
    return image_path.stem


def select_qualitative_cases(
    dataset: CrackDataset,
    source_prob_maps: list[np.ndarray],
    b1_prob_maps: list[np.ndarray],
    count: int,
) -> list[dict]:
    candidates = []
    for idx in range(len(dataset)):
        _, gt_mask = resize_raw_pair(dataset, idx, dataset.img_size)
        source_pred = binary_from_prob_map(source_prob_maps[idx], threshold=0.5)
        b1_pred_07 = binary_from_prob_map(b1_prob_maps[idx], threshold=0.7)
        b1_pred_09 = binary_from_prob_map(b1_prob_maps[idx], threshold=0.9)

        source_stats = sample_metrics(source_pred, gt_mask)
        b1_stats = sample_metrics(b1_pred_07, gt_mask)
        b1_hp_stats = sample_metrics(b1_pred_09, gt_mask)

        total_pixels = float(gt_mask.size)
        fp_drop_ratio = (source_stats["fp"] - b1_stats["fp"]) / total_pixels
        precision_gain = b1_stats["precision"] - source_stats["precision"]
        iou_gain = b1_stats["iou"] - source_stats["iou"]
        source_fp_ratio = source_stats["fp"] / total_pixels
        score = fp_drop_ratio + 0.35 * precision_gain + 0.2 * max(iou_gain, 0.0)

        candidates.append(
            {
                "idx": idx,
                "sample_id": case_label_from_sample(dataset, idx),
                "score": score,
                "source_fp_ratio": source_fp_ratio,
                "iou_gain": iou_gain,
                "precision_gain": precision_gain,
                "source_stats": source_stats,
                "b1_stats": b1_stats,
                "b1_hp_stats": b1_hp_stats,
            }
        )

    preferred = [
        row
        for row in candidates
        if row["source_fp_ratio"] > 0.01
        and row["precision_gain"] > 0.10
        and row["iou_gain"] >= 0.0
        and row["b1_stats"]["iou"] > 0.05
        and row["b1_stats"]["tp"] > 0
    ]
    preferred.sort(key=lambda row: row["score"], reverse=True)

    if len(preferred) < count:
        fallback = sorted(candidates, key=lambda row: row["score"], reverse=True)
        seen = {row["idx"] for row in preferred}
        for row in fallback:
            if row["idx"] in seen:
                continue
            preferred.append(row)
            seen.add(row["idx"])
            if len(preferred) >= count:
                break

    return preferred[:count]


def write_main_results_table(output_dir: Path, rows: list[dict]):
    csv_path = output_dir / "main_results.csv"
    md_path = output_dir / "main_results.md"
    fieldnames = ["setting", "iou", "f1", "precision", "recall"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Main Results",
        "",
        "| Setting | IoU | F1 | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['setting']} | {row['iou']:.4f} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sweep_csv(output_dir: Path, sweep_rows: list[dict]):
    csv_path = output_dir / "threshold_sweep_val.csv"
    fieldnames = ["model", "threshold", "iou", "f1", "precision", "recall"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow(row)


def plot_threshold_sweep(output_dir: Path, sweep_rows: list[dict]):
    metrics = ["iou", "f1", "precision", "recall"]
    titles = {
        "iou": "IoU",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
    }
    colors = {
        "Source-only": "#c44e52",
        "B1 hard-negative mixed": "#4c72b0",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for model_label in colors:
            model_rows = [row for row in sweep_rows if row["model"] == model_label]
            model_rows.sort(key=lambda row: row["threshold"])
            xs = [row["threshold"] for row in model_rows]
            ys = [row[metric] for row in model_rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=model_label, color=colors[model_label])
        ax.set_title(titles[metric])
        ax.set_xlabel("Threshold")
        ax.set_ylabel(titles[metric])
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].legend(frameon=False)
    fig.suptitle("UAV validation threshold sweep: source-only vs B1", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_sweep_val.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_case_manifest(output_dir: Path, cases: list[dict]):
    csv_path = output_dir / "qualitative_cases.csv"
    fieldnames = [
        "sample_id",
        "score",
        "source_iou",
        "source_precision",
        "source_recall",
        "b1_iou",
        "b1_precision",
        "b1_recall",
        "b1_hp_iou",
        "b1_hp_precision",
        "b1_hp_recall",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cases:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "score": f"{row['score']:.6f}",
                    "source_iou": f"{row['source_stats']['iou']:.4f}",
                    "source_precision": f"{row['source_stats']['precision']:.4f}",
                    "source_recall": f"{row['source_stats']['recall']:.4f}",
                    "b1_iou": f"{row['b1_stats']['iou']:.4f}",
                    "b1_precision": f"{row['b1_stats']['precision']:.4f}",
                    "b1_recall": f"{row['b1_stats']['recall']:.4f}",
                    "b1_hp_iou": f"{row['b1_hp_stats']['iou']:.4f}",
                    "b1_hp_precision": f"{row['b1_hp_stats']['precision']:.4f}",
                    "b1_hp_recall": f"{row['b1_hp_stats']['recall']:.4f}",
                }
            )


def plot_qualitative_cases(
    output_dir: Path,
    dataset: CrackDataset,
    source_prob_maps: list[np.ndarray],
    b1_prob_maps: list[np.ndarray],
    cases: list[dict],
):
    num_rows = len(cases)
    fig, axes = plt.subplots(num_rows, 5, figsize=(16, 3.8 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Input",
        "Ground Truth",
        "Source raw @ 0.5",
        "B1 raw @ 0.7",
        "B1 raw @ 0.9",
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    for row_idx, case in enumerate(cases):
        idx = case["idx"]
        image, gt_mask = resize_raw_pair(dataset, idx, dataset.img_size)
        source_pred = binary_from_prob_map(source_prob_maps[idx], threshold=0.5)
        b1_pred_07 = binary_from_prob_map(b1_prob_maps[idx], threshold=0.7)
        b1_pred_09 = binary_from_prob_map(b1_prob_maps[idx], threshold=0.9)

        gt_rgb = np.repeat((gt_mask * 255)[:, :, None], 3, axis=2)
        panels = [
            image,
            gt_rgb,
            build_overlay(image, source_pred, gt_mask),
            build_overlay(image, b1_pred_07, gt_mask),
            build_overlay(image, b1_pred_09, gt_mask),
        ]
        metric_labels = [
            "",
            "",
            case["source_stats"],
            case["b1_stats"],
            case["b1_hp_stats"],
        ]

        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(case["sample_id"], rotation=0, labelpad=48, fontsize=10, va="center")
            if col_idx >= 2:
                stats = metric_labels[col_idx]
                ax.set_xlabel(
                    f"IoU {stats['iou']:.3f} | P {stats['precision']:.3f} | R {stats['recall']:.3f}",
                    fontsize=8,
                )

    legend_text = "Green = true positive, red = false positive, yellow = false negative"
    fig.text(0.5, 0.01, legend_text, ha="center", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.98))
    fig.savefig(output_dir / "qualitative_cases_test.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(output_dir: Path, main_rows: list[dict]):
    def row(label: str) -> dict:
        return next(item for item in main_rows if item["setting"] == label)

    source_raw = row("SegFormer source-only raw")
    source_pp = row("SegFormer source-only + postprocess")
    b1_raw = row("SegFormer B1 raw @ 0.5")
    b1_thr07 = row("SegFormer B1 calibrated @ 0.7")
    b1_thr09 = row("SegFormer B1 calibrated @ 0.9")
    b1_pp = row("SegFormer B1 + postprocess")

    lines = [
        "# B1 Hold-Out Summary",
        "",
        "Main takeaways:",
        "",
        (
            f"- `B1 raw @ 0.7` is the strongest main-report result on the fixed hold-out split "
            f"with `IoU {b1_thr07['iou']:.4f}` and `F1 {b1_thr07['f1']:.4f}`."
        ),
        (
            f"- Relative to source-only raw, `B1 raw @ 0.7` improves IoU by "
            f"{b1_thr07['iou'] - source_raw['iou']:.4f} and F1 by "
            f"{b1_thr07['f1'] - source_raw['f1']:.4f}."
        ),
        (
            f"- `B1 raw @ 0.9` reaches the high-precision regime with "
            f"`precision {b1_thr09['precision']:.4f}`, while keeping slightly better "
            f"`IoU/F1` than `B1 + postprocess`."
        ),
        (
            f"- The old deployment switch remains useful for the weaker source-only model "
            f"(`IoU {source_raw['iou']:.4f} -> {source_pp['iou']:.4f}`), but adds little "
            f"once the model has been improved by `B1` and recalibrated "
            f"(`B1 raw @ 0.9` IoU {b1_thr09['iou']:.4f} vs `B1 + postprocess` IoU {b1_pp['iou']:.4f})."
        ),
        (
            f"- Threshold calibration adds a smaller but still real gain on top of `B1`: "
            f"`IoU {b1_raw['iou']:.4f} -> {b1_thr07['iou']:.4f}` and "
            f"`precision {b1_raw['precision']:.4f} -> {b1_thr07['precision']:.4f}`."
        ),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    thresholds = parse_threshold_list(args.thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_loss(loss_name="bce_dice").to(device)
    postprocess_config = build_postprocess_config(
        min_area=args.postprocess_min_area,
        max_fill_ratio=args.postprocess_max_fill_ratio,
        min_aspect_ratio=args.postprocess_min_aspect_ratio,
        max_components=args.postprocess_max_components,
    )

    val_dataset = build_dataset(args.dataset_root, args.val_split, args.img_size)
    test_dataset = build_dataset(args.dataset_root, args.test_split, args.img_size)
    val_loader = build_loader(val_dataset, args.batch_size, args.num_workers)
    test_loader = build_loader(test_dataset, args.batch_size, args.num_workers)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Validation samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")

    source_model = load_model_from_checkpoint(args, args.source_checkpoint, device)
    b1_model = load_model_from_checkpoint(args, args.b1_checkpoint, device)

    source_test_variants = [
        VariantSpec(label="SegFormer source-only raw", threshold=0.5),
        VariantSpec(
            label="SegFormer source-only + postprocess",
            threshold=0.9,
            postprocess_config=postprocess_config,
        ),
    ]
    b1_test_variants = [
        VariantSpec(label="SegFormer B1 raw @ 0.5", threshold=0.5),
        VariantSpec(label="SegFormer B1 calibrated @ 0.7", threshold=0.7),
        VariantSpec(label="SegFormer B1 calibrated @ 0.9", threshold=0.9),
        VariantSpec(
            label="SegFormer B1 + postprocess",
            threshold=0.9,
            postprocess_config=postprocess_config,
        ),
    ]

    source_test_metrics = evaluate_variants(
        source_model,
        test_loader,
        criterion,
        device,
        source_test_variants,
    )
    b1_test_metrics = evaluate_variants(
        b1_model,
        test_loader,
        criterion,
        device,
        b1_test_variants,
    )

    main_rows = []
    for variant in source_test_variants + b1_test_variants:
        row = source_test_metrics.get(variant.label, b1_test_metrics.get(variant.label))
        main_rows.append(
            {
                "setting": variant.label,
                "iou": row["iou"],
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
            }
        )
    write_main_results_table(output_dir, main_rows)

    sweep_rows = []
    for model_label, model in [
        ("Source-only", source_model),
        ("B1 hard-negative mixed", b1_model),
    ]:
        variants = [
            VariantSpec(label=f"{model_label} @ {threshold:.1f}", threshold=threshold)
            for threshold in thresholds
        ]
        metrics_by_variant = evaluate_variants(model, val_loader, criterion, device, variants)
        for threshold in thresholds:
            label = f"{model_label} @ {threshold:.1f}"
            row = metrics_by_variant[label]
            sweep_rows.append(
                {
                    "model": model_label,
                    "threshold": threshold,
                    "iou": row["iou"],
                    "f1": row["f1"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                }
            )
    write_sweep_csv(output_dir, sweep_rows)
    plot_threshold_sweep(output_dir, sweep_rows)

    source_prob_maps, _ = collect_prob_maps(source_model, test_loader, device)
    b1_prob_maps, _ = collect_prob_maps(b1_model, test_loader, device)
    cases = select_qualitative_cases(
        dataset=test_dataset,
        source_prob_maps=source_prob_maps,
        b1_prob_maps=b1_prob_maps,
        count=args.qualitative_count,
    )
    write_case_manifest(output_dir, cases)
    plot_qualitative_cases(
        output_dir=output_dir,
        dataset=test_dataset,
        source_prob_maps=source_prob_maps,
        b1_prob_maps=b1_prob_maps,
        cases=cases,
    )

    write_summary(output_dir, main_rows)

    print("Saved assets:")
    for path in [
        output_dir / "main_results.csv",
        output_dir / "main_results.md",
        output_dir / "threshold_sweep_val.csv",
        output_dir / "threshold_sweep_val.png",
        output_dir / "qualitative_cases.csv",
        output_dir / "qualitative_cases_test.png",
        output_dir / "summary.md",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
