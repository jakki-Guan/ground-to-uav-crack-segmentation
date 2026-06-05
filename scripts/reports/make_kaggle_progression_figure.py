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
from model import SEGFORMER_B2_MODEL_NAME, get_model


matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class StageSpec:
    key: str
    title: str
    caption_label: str
    model_name: str
    checkpoint_path: str
    threshold: float
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    pretrained_model_name: str = SEGFORMER_B2_MODEL_NAME


DEFAULT_SAMPLE_IDS = ("slide1290", "slide1434", "slide1089")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a Kaggle UAV qualitative progression figure with fixed columns "
            "Input / GT / Source-only / ADVENT / B1 / B2 fs10."
        )
    )
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=list(DEFAULT_SAMPLE_IDS),
        help="Ordered sample ids to show as rows.",
    )
    parser.add_argument(
        "--output-image",
        default="results/report_assets/final_comparison/qualitative_progression_test.png",
    )
    parser.add_argument(
        "--output-metrics",
        default="results/report_assets/final_comparison/qualitative_progression_metrics.csv",
    )
    parser.add_argument(
        "--output-summary",
        default="results/report_assets/final_comparison/qualitative_progression_summary.md",
    )
    parser.add_argument("--source-checkpoint", default="checkpoints/segformer_b2_plain_360.pth")
    parser.add_argument("--source-threshold", type=float, default=0.5)
    parser.add_argument("--advent-checkpoint", default="checkpoints/advent_deeplabv3plus_crack500_to_uav.pth")
    parser.add_argument("--advent-threshold", type=float, default=0.9)
    parser.add_argument(
        "--b1-checkpoint",
        default="checkpoints/segformer_b2_b1_tsbank_thr080_mean082.pth",
    )
    parser.add_argument("--b1-threshold", type=float, default=0.6)
    parser.add_argument("--b2-fs10-checkpoint", default="checkpoints/segformer_b2_b2_fs10_seed42.pth")
    parser.add_argument("--b2-fs10-threshold", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def build_dataset(root: str, split: str, img_size: int) -> CrackDataset:
    return CrackDataset(root=root, split=split, img_size=img_size)


def build_loader(dataset: CrackDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def load_checkpoint_model(stage: StageSpec, device: torch.device):
    model = get_model(
        model_name=stage.model_name,
        encoder_name=stage.encoder_name,
        encoder_weights=stage.encoder_weights,
        in_channels=3,
        classes=1,
        pretrained_model_name=stage.pretrained_model_name,
    )
    state_dict = torch.load(stage.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def collect_prob_maps(model, loader: DataLoader, device: torch.device) -> list[np.ndarray]:
    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            probabilities.extend(probs)
    return probabilities


def resize_raw_pair(dataset: CrackDataset, idx: int, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    image, mask = dataset.get_raw(idx)
    image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized.astype(np.uint8)


def binary_from_prob_map(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_map > threshold).astype(np.uint8)


def build_prediction_overlay(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    tp = np.logical_and(pred_mask == 1, gt_mask == 1)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)

    overlay[tp] = 0.5 * overlay[tp] + 0.5 * np.array([0, 255, 0], dtype=np.float32)
    overlay[fp] = 0.45 * overlay[fp] + 0.55 * np.array([255, 0, 0], dtype=np.float32)
    overlay[fn] = 0.45 * overlay[fn] + 0.55 * np.array([255, 255, 0], dtype=np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


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


def sample_id_from_dataset(dataset: CrackDataset, idx: int) -> str:
    image_path, _ = dataset.samples[idx]
    return Path(image_path).stem


def resolve_sample_indices(dataset: CrackDataset, sample_ids: list[str]) -> list[int]:
    index_by_id = {sample_id_from_dataset(dataset, idx): idx for idx in range(len(dataset))}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise KeyError(f"Sample ids not found in dataset split: {missing}")
    return [index_by_id[sample_id] for sample_id in sample_ids]


def compute_stage_prob_maps(
    stage_specs: list[StageSpec],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, list[np.ndarray]]:
    prob_maps = {}
    for stage in stage_specs:
        model = load_checkpoint_model(stage=stage, device=device)
        prob_maps[stage.key] = collect_prob_maps(model=model, loader=loader, device=device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return prob_maps


def build_case_rows(
    dataset: CrackDataset,
    sample_indices: list[int],
    stage_specs: list[StageSpec],
    stage_prob_maps: dict[str, list[np.ndarray]],
) -> list[dict]:
    rows = []
    for idx in sample_indices:
        sample_id = sample_id_from_dataset(dataset, idx)
        image_path, mask_path = dataset.samples[idx]
        image, gt_mask = resize_raw_pair(dataset, idx, dataset.img_size)
        stage_rows = {}
        for stage in stage_specs:
            pred_mask = binary_from_prob_map(stage_prob_maps[stage.key][idx], stage.threshold)
            stage_rows[stage.key] = {
                "stage": stage,
                "pred_mask": pred_mask,
                "overlay": build_prediction_overlay(image=image, pred_mask=pred_mask, gt_mask=gt_mask),
                "metrics": sample_metrics(pred_mask=pred_mask, gt_mask=gt_mask),
            }
        rows.append(
            {
                "idx": idx,
                "sample_id": sample_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image": image,
                "gt_mask": gt_mask,
                "gt_panel": np.repeat((gt_mask * 255)[:, :, None], 3, axis=2),
                "stages": stage_rows,
            }
        )
    return rows


def render_progression_figure(
    output_path: Path,
    rows: list[dict],
    stage_specs: list[StageSpec],
    dpi: int = 220,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    column_titles = [
        "Input",
        "GT",
        *[stage.title for stage in stage_specs],
    ]
    num_rows = len(rows)
    num_cols = len(column_titles)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3.18 * num_cols, 3.12 * num_rows),
        gridspec_kw={"hspace": 0.06, "wspace": 0.05},
    )
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=11)

    for row_idx, row in enumerate(rows):
        panels = [row["image"], row["gt_panel"]]
        panels.extend(row["stages"][stage.key]["overlay"] for stage in stage_specs)
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"Case {row_idx + 1}", rotation=0, labelpad=34, fontsize=10.5, va="center")
            if col_idx >= 2:
                stage = stage_specs[col_idx - 2]
                stats = row["stages"][stage.key]["metrics"]
                ax.text(
                    0.5,
                    -0.017,
                    f"IoU {stats['iou']:.3f} | P {stats['precision']:.3f} | R {stats['recall']:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.2,
                    clip_on=False,
                )

    fig.text(
        0.5,
        0.002,
        "Green = true positive, red = false positive, yellow = false negative",
        ha="center",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.065, right=0.997, top=0.962, bottom=0.052, hspace=0.06, wspace=0.05)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def write_metrics_csv(output_path: Path, rows: list[dict], stage_specs: list[StageSpec]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "image_path", "mask_path"]
    for stage in stage_specs:
        fieldnames.extend(
            [
                f"{stage.key}_threshold",
                f"{stage.key}_checkpoint",
                f"{stage.key}_iou",
                f"{stage.key}_f1",
                f"{stage.key}_precision",
                f"{stage.key}_recall",
                f"{stage.key}_tp",
                f"{stage.key}_fp",
                f"{stage.key}_fn",
            ]
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
            }
            for stage in stage_specs:
                stats = row["stages"][stage.key]["metrics"]
                record[f"{stage.key}_threshold"] = stage.threshold
                record[f"{stage.key}_checkpoint"] = stage.checkpoint_path
                record[f"{stage.key}_iou"] = f"{stats['iou']:.6f}"
                record[f"{stage.key}_f1"] = f"{stats['f1']:.6f}"
                record[f"{stage.key}_precision"] = f"{stats['precision']:.6f}"
                record[f"{stage.key}_recall"] = f"{stats['recall']:.6f}"
                record[f"{stage.key}_tp"] = int(stats["tp"])
                record[f"{stage.key}_fp"] = int(stats["fp"])
                record[f"{stage.key}_fn"] = int(stats["fn"])
            writer.writerow(record)


def write_summary(output_path: Path, rows: list[dict], stage_specs: list[StageSpec]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    case_map = ", ".join(f"Case {idx + 1} = {row['sample_id']}" for idx, row in enumerate(rows))
    stage_by_key = {stage.key: stage for stage in stage_specs}
    threshold_note = (
        f"Source-only, Selected B1, and B2 fs10 use thresholds "
        f"`{stage_by_key['source_only'].threshold}`, `{stage_by_key['b1_selected'].threshold}`, "
        f"and `{stage_by_key['b2_fs10'].threshold}`, respectively; "
        f"ADVENT-style uses the validation-selected threshold `{stage_by_key['advent'].threshold}`."
    )
    caption_text = (
        "Representative qualitative comparison on the primary Kaggle UAV target. "
        "Columns show the input image, ground-truth mask, source-only prediction, "
        "ADVENT-style UDA prediction, selected B1 prediction, and B2 fs10 prediction. "
        "Green, red, and yellow denote true-positive, false-positive, and false-negative pixels, respectively. "
        "Source-only transfer produces structured false positives along pavement boundaries and crack-like background regions. "
        "ADVENT-style adaptation partially reduces false positives but remains conservative at a high validation-selected threshold, "
        "while selected B1 and B2 fs10 reduce excessive false-positive activation in these examples. "
        "Per-sample ordering may vary; aggregate trends are reported in Table XI."
    )
    lines = [
        "# Kaggle Qualitative Progression",
        "",
        (
            "Fixed Kaggle UAV qualitative comparison for the paper-facing story "
            "`source-only over-activation -> UDA partial relief -> B1 suppression -> B2 recovery`."
        ),
        "",
        f"- Samples: `{', '.join(row['sample_id'] for row in rows)}`",
        f"- Paper row labels: `{case_map}`",
        "- Columns: `Input / GT / Source-only / ADVENT / B1 / B2 fs10`",
        "- Colors: `green = TP`, `red = FP`, `yellow = FN`",
        f"- Threshold note: {threshold_note}",
        "- Backbone note: `ADVENT-style` uses `DeepLabV3+`; the `Source-only`, `Selected B1`, and `B2 fs10` columns use `SegFormer-B2`.",
        "",
        "Paper caption:",
        "",
        f"> {caption_text}",
        "",
        "Stage details:",
        "",
    ]
    for stage in stage_specs:
        lines.append(
            f"- `{stage.caption_label}`: checkpoint `{stage.checkpoint_path}`, threshold `{stage.threshold}`"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    stage_specs = [
        StageSpec(
            key="source_only",
            title="Source-only\nSegFormer-B2",
            caption_label="Source-only SegFormer-B2",
            model_name="SegFormer-B2",
            checkpoint_path=args.source_checkpoint,
            threshold=args.source_threshold,
        ),
        StageSpec(
            key="advent",
            title="ADVENT-style\nDeepLabV3+",
            caption_label="ADVENT-style DeepLabV3+",
            model_name="DeepLabV3Plus",
            checkpoint_path=args.advent_checkpoint,
            threshold=args.advent_threshold,
        ),
        StageSpec(
            key="b1_selected",
            title="Selected B1\nSegFormer-B2",
            caption_label="Selected B1 SegFormer-B2",
            model_name="SegFormer-B2",
            checkpoint_path=args.b1_checkpoint,
            threshold=args.b1_threshold,
        ),
        StageSpec(
            key="b2_fs10",
            title="B2 fs10\nSegFormer-B2",
            caption_label="B2 fs10 SegFormer-B2",
            model_name="SegFormer-B2",
            checkpoint_path=args.b2_fs10_checkpoint,
            threshold=args.b2_fs10_threshold,
        ),
    ]

    dataset = build_dataset(root=args.dataset_root, split=args.split, img_size=args.img_size)
    loader = build_loader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    sample_indices = resolve_sample_indices(dataset=dataset, sample_ids=args.sample_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_prob_maps = compute_stage_prob_maps(stage_specs=stage_specs, loader=loader, device=device)
    rows = build_case_rows(
        dataset=dataset,
        sample_indices=sample_indices,
        stage_specs=stage_specs,
        stage_prob_maps=stage_prob_maps,
    )

    render_progression_figure(
        output_path=Path(args.output_image),
        rows=rows,
        stage_specs=stage_specs,
        dpi=args.dpi,
    )
    write_metrics_csv(output_path=Path(args.output_metrics), rows=rows, stage_specs=stage_specs)
    write_summary(output_path=Path(args.output_summary), rows=rows, stage_specs=stage_specs)


if __name__ == "__main__":
    main()
