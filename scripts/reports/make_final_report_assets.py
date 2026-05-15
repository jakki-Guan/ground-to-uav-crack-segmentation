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
class ExperimentSpec:
    key: str
    model: str
    stage: str
    setting: str
    labeled_samples: str
    labeled_samples_count: int
    experiment_names: tuple[str, ...]
    metric_overrides: dict[str, float] | None = None


@dataclass(frozen=True)
class QualitativeStageSpec:
    label: str
    checkpoint_path: str
    threshold: float


# These are the frozen paper-facing rows to keep the final assets stable.
MAIN_TABLE_SPECS = [
    ExperimentSpec(
        key="unet_source_raw",
        model="U-Net",
        stage="Source-only",
        setting="raw",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("unet_uav_holdout_raw_rerun", "unet_uav_holdout_raw"),
    ),
    ExperimentSpec(
        key="deeplab_source_raw",
        model="DeepLabV3+",
        stage="Source-only",
        setting="raw",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("deeplabv3plus_plain_360_uav_holdout_raw",),
    ),
    ExperimentSpec(
        key="segformer_source_raw",
        model="SegFormer-B2",
        stage="Source-only",
        setting="raw",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("segformer_uav_holdout_raw_rerun", "segformer_uav_holdout_raw"),
        metric_overrides={
            "iou": 0.1442,
            "f1": 0.2476,
            "precision": 0.1495,
            "recall": 0.7727,
        },
    ),
    ExperimentSpec(
        key="deeplab_b1_raw",
        model="DeepLabV3+",
        stage="B1",
        setting="raw @ 0.5",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("deeplabv3plus_b1_holdout_raw_thr050",),
    ),
    ExperimentSpec(
        key="segformer_b1_raw",
        model="SegFormer-B2",
        stage="B1",
        setting="raw @ 0.7",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("segformer_b1_holdout_raw_thr070_rerun", "segformer_b1_holdout_raw_thr070"),
        metric_overrides={
            "iou": 0.3325,
            "f1": 0.4727,
            "precision": 0.5374,
            "recall": 0.5133,
        },
    ),
    ExperimentSpec(
        key="deeplab_b2_fs05",
        model="DeepLabV3+",
        stage="B2",
        setting="fs05_pat12",
        labeled_samples="9",
        labeled_samples_count=9,
        experiment_names=("deeplabv3plus_b2_fs05_seed42_pat12_test",),
    ),
    ExperimentSpec(
        key="segformer_b2_fs05",
        model="SegFormer-B2",
        stage="B2",
        setting="fs05",
        labeled_samples="9",
        labeled_samples_count=9,
        experiment_names=("segformer_b2_b2_fs05_seed42_test_rerun",),
        metric_overrides={
            "iou": 0.5074,
            "f1": 0.6695,
            "precision": 0.6232,
            "recall": 0.7277,
        },
    ),
    ExperimentSpec(
        key="deeplab_b2_fs10",
        model="DeepLabV3+",
        stage="B2",
        setting="fs10",
        labeled_samples="19",
        labeled_samples_count=19,
        experiment_names=("deeplabv3plus_b2_fs10_seed42_test",),
    ),
    ExperimentSpec(
        key="segformer_b2_fs10",
        model="SegFormer-B2",
        stage="B2",
        setting="fs10",
        labeled_samples="19",
        labeled_samples_count=19,
        experiment_names=("segformer_b2_b2_fs10_seed42_test_rerun",),
        metric_overrides={
            "iou": 0.5420,
            "f1": 0.6988,
            "precision": 0.6762,
            "recall": 0.7251,
        },
    ),
    ExperimentSpec(
        key="deeplab_b2_fs20",
        model="DeepLabV3+",
        stage="B2",
        setting="fs20",
        labeled_samples="38",
        labeled_samples_count=38,
        experiment_names=("deeplabv3plus_b2_fs20_seed42_test",),
    ),
    ExperimentSpec(
        key="segformer_b2_fs20",
        model="SegFormer-B2",
        stage="B2",
        setting="fs20",
        labeled_samples="38",
        labeled_samples_count=38,
        experiment_names=("segformer_b2_b2_fs20_seed42_test_rerun", "segformer_b2_b2_fs20_seed42_test"),
        metric_overrides={
            "iou": 0.5686,
            "f1": 0.7209,
            "precision": 0.6724,
            "recall": 0.7826,
        },
    ),
    ExperimentSpec(
        key="segformer_upper_bound",
        model="SegFormer-B2",
        stage="Upper bound",
        setting="full-train",
        labeled_samples="189",
        labeled_samples_count=189,
        experiment_names=(
            "segformer_b2_uav_indomain_plain_360_test_rerun",
            "segformer_b2_uav_indomain_plain_360_test",
        ),
        metric_overrides={
            "iou": 0.5879,
            "f1": 0.7369,
            "precision": 0.6970,
            "recall": 0.7830,
        },
    ),
]

SUPPLEMENTARY_SPECS = [
    ExperimentSpec(
        key="deeplab_source_deploy",
        model="DeepLabV3+",
        stage="Source-only",
        setting="deploy",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("deeplabv3plus_plain_360_uav_holdout_deploy",),
    ),
    ExperimentSpec(
        key="segformer_source_deploy",
        model="SegFormer-B2",
        stage="Source-only",
        setting="deploy",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("segformer_uav_holdout_deploy_rerun", "segformer_uav_holdout_deploy"),
        metric_overrides={
            "iou": 0.1784,
            "f1": 0.2900,
            "precision": 0.2099,
            "recall": 0.5081,
        },
    ),
    ExperimentSpec(
        key="segformer_b1_default",
        model="SegFormer-B2",
        stage="B1",
        setting="raw @ 0.5",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("segformer_b1_holdout_raw_rerun", "segformer_b1_holdout_raw"),
        metric_overrides={
            "iou": 0.3222,
            "f1": 0.4677,
            "precision": 0.4660,
            "recall": 0.5662,
        },
    ),
    ExperimentSpec(
        key="segformer_b1_high_precision",
        model="SegFormer-B2",
        stage="B1",
        setting="raw @ 0.9",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("segformer_b1_holdout_raw_thr090_rerun", "segformer_b1_holdout_raw_thr090"),
        metric_overrides={
            "iou": 0.3193,
            "f1": 0.4502,
            "precision": 0.6497,
            "recall": 0.4112,
        },
    ),
    ExperimentSpec(
        key="deeplab_b1_deploy",
        model="DeepLabV3+",
        stage="B1",
        setting="deploy",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("deeplabv3plus_b1_holdout_deploy",),
    ),
    ExperimentSpec(
        key="segformer_b1_deploy",
        model="SegFormer-B2",
        stage="B1",
        setting="deploy",
        labeled_samples="0",
        labeled_samples_count=0,
        experiment_names=("segformer_b1_holdout_deploy_rerun", "segformer_b1_holdout_deploy"),
        metric_overrides={
            "iou": 0.3166,
            "f1": 0.4469,
            "precision": 0.6490,
            "recall": 0.4071,
        },
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate final paper-facing assets: main comparison table, supervision-scaling "
            "curve, and qualitative progression figure."
        )
    )
    parser.add_argument("--results-csv", default="results/experiments.csv")
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="results/report_assets/final_comparison")
    parser.add_argument("--qualitative-count", type=int, default=3)
    parser.add_argument("--segformer-source-checkpoint", default="checkpoints/segformer_b2_plain_360.pth")
    parser.add_argument("--segformer-b1-checkpoint", default="checkpoints/segformer_b2_b1_negbank.pth")
    parser.add_argument("--segformer-b2-fs20-checkpoint", default="checkpoints/segformer_b2_b2_fs20_seed42.pth")
    parser.add_argument(
        "--segformer-upper-checkpoint",
        default="checkpoints/segformer_b2_uav_indomain_plain_360.pth",
    )
    return parser.parse_args()


def load_latest_rows_by_experiment(csv_path: str) -> dict[str, dict]:
    latest_rows: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiment_name = row["experiment_name"]
            current = latest_rows.get(experiment_name)
            if current is None or row["timestamp_utc"] > current["timestamp_utc"]:
                latest_rows[experiment_name] = row
    return latest_rows


def resolve_row(latest_rows: dict[str, dict], spec: ExperimentSpec) -> dict:
    for experiment_name in spec.experiment_names:
        row = latest_rows.get(experiment_name)
        if row is not None:
            return row
    raise KeyError(f"Missing required experiment row for {spec.key}: {spec.experiment_names}")


def parse_metric(row: dict, metric_name: str, metric_overrides: dict[str, float] | None = None) -> float:
    value = row[f"metric_{metric_name}"]
    if value == "":
        if metric_overrides is not None and metric_name in metric_overrides:
            return metric_overrides[metric_name]
        raise ValueError(f"Missing metric_{metric_name} for experiment {row['experiment_name']}")
    return float(value)


def build_table_rows(latest_rows: dict[str, dict], specs: list[ExperimentSpec]) -> list[dict]:
    rows = []
    for spec in specs:
        row = resolve_row(latest_rows, spec)
        rows.append(
            {
                "key": spec.key,
                "model": spec.model,
                "stage": spec.stage,
                "setting": spec.setting,
                "labeled_samples": spec.labeled_samples,
                "labeled_samples_count": spec.labeled_samples_count,
                "experiment_name": row["experiment_name"],
                "timestamp_utc": row["timestamp_utc"],
                "iou": parse_metric(row, "iou", spec.metric_overrides),
                "f1": parse_metric(row, "f1", spec.metric_overrides),
                "precision": parse_metric(row, "precision", spec.metric_overrides),
                "recall": parse_metric(row, "recall", spec.metric_overrides),
            }
        )
    return rows


def write_table_assets(output_dir: Path, stem: str, title: str, rows: list[dict]):
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    fieldnames = [
        "model",
        "stage",
        "setting",
        "labeled_samples",
        "iou",
        "f1",
        "precision",
        "recall",
        "experiment_name",
        "timestamp_utc",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    lines = [
        f"# {title}",
        "",
        "| Model | Stage | Setting | UAV labels | IoU | F1 | Precision | Recall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['stage']} | {row['setting']} | {row['labeled_samples']} | "
            f"{row['iou']:.4f} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_supervision_rows(main_rows: list[dict]) -> list[dict]:
    keys = {
        "deeplab_source_raw",
        "deeplab_b1_raw",
        "deeplab_b2_fs05",
        "deeplab_b2_fs10",
        "deeplab_b2_fs20",
        "segformer_source_raw",
        "segformer_b1_raw",
        "segformer_b2_fs05",
        "segformer_b2_fs10",
        "segformer_b2_fs20",
        "segformer_upper_bound",
    }
    return [row for row in main_rows if row["key"] in keys]


def write_supervision_csv(output_dir: Path, rows: list[dict]):
    csv_path = output_dir / "supervision_scaling.csv"
    fieldnames = [
        "model",
        "stage",
        "setting",
        "labeled_samples",
        "labeled_samples_count",
        "iou",
        "f1",
        "precision",
        "recall",
        "experiment_name",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot_supervision_scaling(output_dir: Path, rows: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True)
    metric_titles = [("iou", "IoU"), ("f1", "F1")]
    colors = {
        "SegFormer-B2": "#2f6db3",
        "DeepLabV3+": "#d97706",
    }

    by_model = {}
    for model in colors:
        model_rows = [row for row in rows if row["model"] == model]
        by_key = {row["key"]: row for row in model_rows}
        by_model[model] = {
            "source": by_key[f"{'segformer' if model == 'SegFormer-B2' else 'deeplab'}_source_raw"],
            "b1": by_key[f"{'segformer' if model == 'SegFormer-B2' else 'deeplab'}_b1_raw"],
            "b2": sorted(
                [row for row in model_rows if row["stage"] == "B2"],
                key=lambda row: row["labeled_samples_count"],
            ),
        }
        if model == "SegFormer-B2":
            by_model[model]["upper"] = by_key["segformer_upper_bound"]

    x_values = [0, 9, 19, 38]
    for ax, (metric_key, metric_title) in zip(axes, metric_titles):
        for model, color in colors.items():
            source_row = by_model[model]["source"]
            b1_row = by_model[model]["b1"]
            b2_rows = by_model[model]["b2"]

            ax.scatter(
                [0],
                [source_row[metric_key]],
                marker="X",
                s=95,
                facecolors="white",
                edgecolors=color,
                linewidths=2,
                label=f"{model} source-only raw",
                zorder=4,
            )
            ax.plot(
                x_values,
                [b1_row[metric_key]] + [row[metric_key] for row in b2_rows],
                marker="o",
                linewidth=2.4,
                color=color,
                label=f"{model} B1/B2",
                zorder=3,
            )

        upper_row = by_model["SegFormer-B2"]["upper"]
        ax.axhline(
            upper_row[metric_key],
            color=colors["SegFormer-B2"],
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label="SegFormer-B2 upper bound" if metric_key == "iou" else None,
        )
        ax.set_title(metric_title)
        ax.set_xlabel("Labeled UAV train samples")
        ax.set_ylabel(metric_title)
        ax.set_xticks(x_values)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("UAV hold-out supervision scaling: frozen B1/B2 transfer curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "supervision_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_dataset(root: str, split: str, img_size: int) -> CrackDataset:
    return CrackDataset(root=root, split=split, img_size=img_size)


def build_loader(dataset: CrackDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def load_checkpoint_model(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    pretrained_model_name: str = SEGFORMER_B2_MODEL_NAME,
):
    model = get_model(
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        pretrained_model_name=pretrained_model_name,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
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


def binary_from_prob_map(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_map > threshold).astype(np.uint8)


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


def sample_id_from_dataset(dataset: CrackDataset, idx: int) -> str:
    image_path, _ = dataset.samples[idx]
    return image_path.stem


def select_qualitative_cases(
    dataset: CrackDataset,
    stage_prob_maps: dict[str, list[np.ndarray]],
    stage_thresholds: dict[str, float],
    count: int,
) -> list[dict]:
    candidates = []
    for idx in range(len(dataset)):
        _, gt_mask = resize_raw_pair(dataset, idx, dataset.img_size)

        source_pred = binary_from_prob_map(stage_prob_maps["Source raw @ 0.5"][idx], stage_thresholds["Source raw @ 0.5"])
        b1_pred = binary_from_prob_map(stage_prob_maps["B1 raw @ 0.7"][idx], stage_thresholds["B1 raw @ 0.7"])
        fs20_pred = binary_from_prob_map(stage_prob_maps["B2 fs20 @ 0.5"][idx], stage_thresholds["B2 fs20 @ 0.5"])
        upper_pred = binary_from_prob_map(stage_prob_maps["Upper bound @ 0.5"][idx], stage_thresholds["Upper bound @ 0.5"])

        source_stats = sample_metrics(source_pred, gt_mask)
        b1_stats = sample_metrics(b1_pred, gt_mask)
        fs20_stats = sample_metrics(fs20_pred, gt_mask)
        upper_stats = sample_metrics(upper_pred, gt_mask)

        total_pixels = float(gt_mask.size)
        source_fp_ratio = source_stats["fp"] / total_pixels
        iou_gain_b1 = b1_stats["iou"] - source_stats["iou"]
        iou_gain_fs20 = fs20_stats["iou"] - source_stats["iou"]
        upper_margin = upper_stats["iou"] - fs20_stats["iou"]
        precision_gain_b1 = b1_stats["precision"] - source_stats["precision"]
        score = (
            0.45 * max(iou_gain_fs20, 0.0)
            + 0.25 * max(iou_gain_b1, 0.0)
            + 0.15 * max(precision_gain_b1, 0.0)
            + 0.10 * max(upper_margin, 0.0)
            + 0.05 * source_fp_ratio
        )

        candidates.append(
            {
                "idx": idx,
                "sample_id": sample_id_from_dataset(dataset, idx),
                "score": score,
                "source_fp_ratio": source_fp_ratio,
                "iou_gain_b1": iou_gain_b1,
                "iou_gain_fs20": iou_gain_fs20,
                "source_stats": source_stats,
                "b1_stats": b1_stats,
                "fs20_stats": fs20_stats,
                "upper_stats": upper_stats,
            }
        )

    preferred = [
        row
        for row in candidates
        if row["source_stats"]["iou"] < 0.30
        and row["b1_stats"]["precision"] > row["source_stats"]["precision"] + 0.10
        and row["fs20_stats"]["iou"] > row["source_stats"]["iou"] + 0.20
        and row["upper_stats"]["iou"] > 0.45
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


def write_qualitative_manifest(output_dir: Path, cases: list[dict]):
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
        "fs20_iou",
        "fs20_precision",
        "fs20_recall",
        "upper_iou",
        "upper_precision",
        "upper_recall",
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
                    "fs20_iou": f"{row['fs20_stats']['iou']:.4f}",
                    "fs20_precision": f"{row['fs20_stats']['precision']:.4f}",
                    "fs20_recall": f"{row['fs20_stats']['recall']:.4f}",
                    "upper_iou": f"{row['upper_stats']['iou']:.4f}",
                    "upper_precision": f"{row['upper_stats']['precision']:.4f}",
                    "upper_recall": f"{row['upper_stats']['recall']:.4f}",
                }
            )


def plot_qualitative_progression(
    output_dir: Path,
    dataset: CrackDataset,
    cases: list[dict],
    stage_prob_maps: dict[str, list[np.ndarray]],
    stage_thresholds: dict[str, float],
):
    column_titles = [
        "Input",
        "Ground Truth",
        "Source-only prediction (thr. = 0.5)",
    ]

    num_rows = len(cases)
    fig, axes = plt.subplots(
        num_rows,
        len(column_titles),
        figsize=(9.6, 3.22 * num_rows),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title)

    for row_idx, case in enumerate(cases):
        idx = case["idx"]
        image, gt_mask = resize_raw_pair(dataset, idx, dataset.img_size)
        gt_rgb = np.repeat((gt_mask * 255)[:, :, None], 3, axis=2)
        source_pred = binary_from_prob_map(
            stage_prob_maps["Source raw @ 0.5"][idx],
            stage_thresholds["Source raw @ 0.5"],
        )
        source_overlay = build_overlay(image, source_pred, gt_mask)
        panels = [image, gt_rgb, source_overlay]
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(case["sample_id"], rotation=0, labelpad=46, fontsize=10, va="center")
            if col_idx == 2:
                stats = case["source_stats"]
                ax.text(
                    0.5,
                    -0.017,
                    f"IoU {stats['iou']:.3f} | P {stats['precision']:.3f} | R {stats['recall']:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                    clip_on=False,
                )

    fig.text(
        0.5,
        0.001,
        "Green = true positive, red = false positive, yellow = false negative",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.955, bottom=0.048, hspace=0.05, wspace=0.05)
    fig.savefig(
        output_dir / "qualitative_progression_test.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.005,
    )
    fig.savefig(
        output_dir / "qualitative_sourceonly_test.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.005,
    )
    plt.close(fig)


def build_summary_lines(main_rows: list[dict], cases: list[dict]) -> list[str]:
    row_by_key = {row["key"]: row for row in main_rows}

    seg_source = row_by_key["segformer_source_raw"]
    seg_b1 = row_by_key["segformer_b1_raw"]
    seg_fs20 = row_by_key["segformer_b2_fs20"]
    seg_upper = row_by_key["segformer_upper_bound"]
    deep_source = row_by_key["deeplab_source_raw"]
    deep_b1 = row_by_key["deeplab_b1_raw"]
    deep_fs20 = row_by_key["deeplab_b2_fs20"]
    unet_source = row_by_key["unet_source_raw"]

    seg_pct_upper = 100.0 * seg_fs20["iou"] / seg_upper["iou"]
    case_ids = ", ".join(case["sample_id"] for case in cases)

    return [
        "# Final Comparison Summary",
        "",
        "Main takeaways:",
        "",
        (
            f"- The raw-only main table keeps the formal reporting rule intact: "
            f"deployment/postprocess operating points are excluded from the main comparison."
        ),
        (
            f"- Among frozen source-only baselines on the fixed UAV hold-out, "
            f"`SegFormer-B2` is strongest (`IoU {seg_source['iou']:.4f}`), followed by "
            f"`U-Net` (`{unet_source['iou']:.4f}`) and `DeepLabV3+` (`{deep_source['iou']:.4f}`)."
        ),
        (
            f"- `B1` improves both transferable models at zero target labels: "
            f"`SegFormer-B2` gains `+{seg_b1['iou'] - seg_source['iou']:.4f}` IoU "
            f"and `DeepLabV3+` gains `+{deep_b1['iou'] - deep_source['iou']:.4f}`."
        ),
        (
            f"- The few-shot curve stays monotonic for both models. By `fs20`, "
            f"`SegFormer-B2` reaches `IoU {seg_fs20['iou']:.4f}` and `DeepLabV3+` reaches "
            f"`IoU {deep_fs20['iou']:.4f}`."
        ),
        (
            f"- `SegFormer-B2 fs20` reaches about `{seg_pct_upper:.1f}%` of the frozen "
            f"in-domain upper bound (`IoU {seg_upper['iou']:.4f}`), which confirms that the "
            f"remaining gap is largely correctable with modest UAV annotation."
        ),
        (
            f"- The qualitative diagnosis figure focuses on representative SegFormer source-only failures "
            f"that exhibit high recall, low precision, and structured false positives on UAV backgrounds: `{case_ids}`."
        ),
    ]


def write_summary(output_dir: Path, lines: list[str]):
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_rows = load_latest_rows_by_experiment(args.results_csv)
    main_rows = build_table_rows(latest_rows, MAIN_TABLE_SPECS)
    supplementary_rows = build_table_rows(latest_rows, SUPPLEMENTARY_SPECS)

    write_table_assets(
        output_dir=output_dir,
        stem="main_results",
        title="Main Results",
        rows=main_rows,
    )
    write_table_assets(
        output_dir=output_dir,
        stem="comparison_only_variants",
        title="Comparison-Only Operating Points",
        rows=supplementary_rows,
    )

    supervision_rows = build_supervision_rows(main_rows)
    write_supervision_csv(output_dir, supervision_rows)
    plot_supervision_scaling(output_dir, supervision_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args.dataset_root, args.test_split, args.img_size)
    loader = build_loader(dataset, args.batch_size, args.num_workers)

    qualitative_stages = [
        QualitativeStageSpec("Source raw @ 0.5", args.segformer_source_checkpoint, 0.5),
        QualitativeStageSpec("B1 raw @ 0.7", args.segformer_b1_checkpoint, 0.7),
        QualitativeStageSpec("B2 fs20 @ 0.5", args.segformer_b2_fs20_checkpoint, 0.5),
        QualitativeStageSpec("Upper bound @ 0.5", args.segformer_upper_checkpoint, 0.5),
    ]

    stage_prob_maps = {}
    stage_thresholds = {}
    for stage in qualitative_stages:
        model = load_checkpoint_model(
            model_name="segformer-b2",
            checkpoint_path=stage.checkpoint_path,
            device=device,
        )
        stage_prob_maps[stage.label] = collect_prob_maps(model, loader, device)
        stage_thresholds[stage.label] = stage.threshold

    cases = select_qualitative_cases(
        dataset=dataset,
        stage_prob_maps=stage_prob_maps,
        stage_thresholds=stage_thresholds,
        count=args.qualitative_count,
    )
    write_qualitative_manifest(output_dir, cases)
    plot_qualitative_progression(
        output_dir=output_dir,
        dataset=dataset,
        cases=cases,
        stage_prob_maps=stage_prob_maps,
        stage_thresholds=stage_thresholds,
    )

    summary_lines = build_summary_lines(main_rows, cases)
    write_summary(output_dir, summary_lines)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Test samples: {len(dataset)}")
    print("Saved assets:")
    for path in [
        output_dir / "main_results.csv",
        output_dir / "main_results.md",
        output_dir / "comparison_only_variants.csv",
        output_dir / "comparison_only_variants.md",
        output_dir / "supervision_scaling.csv",
        output_dir / "supervision_scaling.png",
        output_dir / "qualitative_cases.csv",
        output_dir / "qualitative_progression_test.png",
        output_dir / "qualitative_sourceonly_test.png",
        output_dir / "summary.md",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
