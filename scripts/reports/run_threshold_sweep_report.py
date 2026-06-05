import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import CrackDataset
from experiment_logger import append_experiment_record, build_experiment_record
from loss import build_loss
from metrics import confusion_stats
from model import SEGFORMER_B2_MODEL_NAME, get_model
from postprocess import build_postprocess_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select a raw prediction threshold on the validation split and report "
            "the frozen threshold on the test split."
        )
    )
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--model-name", default="DeepLabV3Plus")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--pretrained-model-name", default=SEGFORMER_B2_MODEL_NAME)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selection-metric", choices=["iou", "f1"], default="iou")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--results-csv", default="results/experiments.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--loss-name",
        choices=["bce_dice", "tversky", "focal_tversky"],
        default="bce_dice",
    )
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--tversky-gamma", type=float, default=1.33)
    parser.add_argument("--postprocess-min-area", type=int, default=0)
    parser.add_argument("--postprocess-max-fill-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-min-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-max-components", type=int, default=0)
    parser.add_argument("--no-log-test-result", action="store_true")
    return parser.parse_args()


def parse_threshold_list(raw: str) -> list[float]:
    thresholds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            thresholds.append(float(part))
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")
    return sorted(set(thresholds))


def normalize_encoder_weights(encoder_weights: str | None):
    if encoder_weights is None:
        return None
    if encoder_weights.strip().lower() in {"none", "null", ""}:
        return None
    return encoder_weights


def build_loader(args, split: str):
    dataset = CrackDataset(root=args.dataset_root, split=split, img_size=args.img_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return dataset, loader


def evaluate(model, loader, criterion, device, dataset_size, threshold, postprocess_config=None):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            batch_size_now = images.size(0)
            stats = confusion_stats(
                outputs,
                masks,
                threshold=threshold,
                postprocess_config=postprocess_config,
            )
            total_loss += loss.item() * batch_size_now
            total_iou += stats["iou"] * batch_size_now
            total_f1 += stats["f1"] * batch_size_now
            total_precision += stats["precision"] * batch_size_now
            total_recall += stats["recall"] * batch_size_now

    return {
        "loss": total_loss / dataset_size,
        "iou": total_iou / dataset_size,
        "f1": total_f1 / dataset_size,
        "precision": total_precision / dataset_size,
        "recall": total_recall / dataset_size,
    }


def write_csv(path: Path, rows: list[dict]):
    fieldnames = ["split", "threshold", "loss", "iou", "f1", "precision", "recall"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(path: Path, args, selected_row: dict, test_row: dict):
    lines = [
        "# Threshold Selection Report",
        "",
        f"- Checkpoint: `{args.checkpoint_path}`",
        f"- Validation split: `{args.dataset_root}/{args.val_split}`",
        f"- Test split: `{args.dataset_root}/{args.test_split}`",
        f"- Selection metric: `{args.selection_metric}`",
        f"- Selected threshold: `{selected_row['threshold']:.2f}`",
        "",
        "## Validation Selection",
        "",
        (
            f"- IoU `{selected_row['iou']:.4f}`, F1 `{selected_row['f1']:.4f}`, "
            f"precision `{selected_row['precision']:.4f}`, recall `{selected_row['recall']:.4f}`"
        ),
        "",
        "## Confirmatory Test",
        "",
        (
            f"- IoU `{test_row['iou']:.4f}`, F1 `{test_row['f1']:.4f}`, "
            f"precision `{test_row['precision']:.4f}`, recall `{test_row['recall']:.4f}`"
        ),
        "",
        "Threshold selection used validation data only. The test split was evaluated once with the frozen selected threshold.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    thresholds = parse_threshold_list(args.thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset, val_loader = build_loader(args, args.val_split)
    test_dataset, test_loader = build_loader(args, args.test_split)

    model = get_model(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=normalize_encoder_weights(args.encoder_weights),
        in_channels=3,
        classes=1,
        pretrained_model_name=args.pretrained_model_name,
    ).to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    criterion = build_loss(
        loss_name=args.loss_name,
        pos_weight=args.pos_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
    )
    postprocess_config = build_postprocess_config(
        min_area=args.postprocess_min_area,
        max_fill_ratio=args.postprocess_max_fill_ratio,
        min_aspect_ratio=args.postprocess_min_aspect_ratio,
        max_components=args.postprocess_max_components,
    )

    print(f"Using device: {device}")
    print(f"Validation split: {args.dataset_root}/{args.val_split} ({len(val_dataset)} samples)")
    print(f"Test split: {args.dataset_root}/{args.test_split} ({len(test_dataset)} samples)")
    print(f"Thresholds: {', '.join(f'{threshold:.2f}' for threshold in thresholds)}")

    val_rows = []
    for threshold in thresholds:
        metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            dataset_size=len(val_dataset),
            threshold=threshold,
            postprocess_config=postprocess_config,
        )
        row = {"split": args.val_split, "threshold": threshold, **metrics}
        val_rows.append(row)
        print(
            f"Val threshold={threshold:.2f} | "
            f"iou={metrics['iou']:.4f} | f1={metrics['f1']:.4f} | "
            f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f}"
        )

    selected_row = max(
        val_rows,
        key=lambda row: (
            row[args.selection_metric],
            row["f1"],
            row["precision"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    selected_threshold = float(selected_row["threshold"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        dataset_size=len(test_dataset),
        threshold=selected_threshold,
        postprocess_config=postprocess_config,
    )
    test_row = {"split": args.test_split, "threshold": selected_threshold, **test_metrics}

    write_csv(output_dir / "threshold_sweep_val.csv", val_rows)
    write_csv(output_dir / "selected_threshold_test.csv", [test_row])
    selection = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "dataset_root": args.dataset_root,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "selection_metric": args.selection_metric,
        "threshold_candidates": thresholds,
        "selected_threshold": selected_threshold,
        "selected_val_metrics": {
            key: selected_row[key] for key in ["loss", "iou", "f1", "precision", "recall"]
        },
        "test_metrics": test_metrics,
    }
    (output_dir / "selected_threshold.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.md", args, selected_row, test_row)

    print(
        f"Selected threshold={selected_threshold:.2f} on val "
        f"({args.selection_metric}={selected_row[args.selection_metric]:.4f})"
    )
    print(
        f"Test @ {selected_threshold:.2f} | "
        f"iou={test_metrics['iou']:.4f} | f1={test_metrics['f1']:.4f} | "
        f"precision={test_metrics['precision']:.4f} | recall={test_metrics['recall']:.4f}"
    )
    print(f"Wrote report assets to {output_dir}")

    if not args.no_log_test_result:
        args.eval_threshold = selected_threshold
        record = build_experiment_record(
            args=args,
            script_name="run_threshold_sweep_report.py",
            stage="test",
            split=args.test_split,
            checkpoint_path=args.checkpoint_path,
            dataset_size=len(test_dataset),
            metrics=test_metrics,
        )
        append_experiment_record(args.results_csv, record)
        print(f"Logged selected-threshold test result to {args.results_csv}")


if __name__ == "__main__":
    main()
