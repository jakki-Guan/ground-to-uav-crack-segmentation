import csv
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_FIELDNAMES = [
    "timestamp_utc",
    "run_id",
    "script",
    "stage",
    "split",
    "experiment_name",
    "dataset_root",
    "train_split",
    "val_dataset_root",
    "val_split",
    "model_name",
    "encoder_name",
    "encoder_weights",
    "pretrained_model_name",
    "img_size",
    "batch_size",
    "epochs",
    "lr",
    "num_workers",
    "seed",
    "augmentation_profile",
    "foreground_crop_prob",
    "foreground_crop_min_scale",
    "foreground_crop_max_scale",
    "aux_negative_root",
    "aux_negative_split",
    "aux_negative_repeat",
    "aux_negative_augmentation_profile",
    "loss_name",
    "pos_weight",
    "tversky_alpha",
    "tversky_beta",
    "tversky_gamma",
    "checkpoint_path",
    "init_checkpoint_path",
    "best_epoch",
    "dataset_size",
    "eval_threshold",
    "postprocess_min_area",
    "postprocess_max_fill_ratio",
    "postprocess_min_aspect_ratio",
    "postprocess_max_components",
    "metric_loss",
    "metric_iou",
    "metric_f1",
    "metric_precision",
    "metric_recall",
    "command",
]


def _utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _current_command():
    return " ".join(shlex.quote(arg) for arg in sys.argv)


def derive_experiment_name(args, checkpoint_path: str):
    explicit_name = getattr(args, "experiment_name", None)
    if explicit_name:
        return explicit_name
    return Path(checkpoint_path).stem


def build_experiment_record(
    args,
    script_name: str,
    stage: str,
    split: str,
    checkpoint_path: str,
    dataset_size: int,
    metrics: dict,
    best_epoch: int | None = None,
):
    timestamp = _utc_timestamp()
    return {
        "timestamp_utc": timestamp,
        "run_id": f"{timestamp}_{os.getpid()}",
        "script": script_name,
        "stage": stage,
        "split": split,
        "experiment_name": derive_experiment_name(args, checkpoint_path),
        "dataset_root": getattr(args, "dataset_root", ""),
        "train_split": getattr(args, "train_split", ""),
        "val_dataset_root": getattr(args, "val_dataset_root", ""),
        "val_split": getattr(args, "val_split", ""),
        "model_name": getattr(args, "model_name", ""),
        "encoder_name": getattr(args, "encoder_name", ""),
        "encoder_weights": getattr(args, "encoder_weights", ""),
        "pretrained_model_name": getattr(args, "pretrained_model_name", ""),
        "img_size": getattr(args, "img_size", ""),
        "batch_size": getattr(args, "batch_size", ""),
        "epochs": getattr(args, "epochs", ""),
        "lr": getattr(args, "lr", ""),
        "num_workers": getattr(args, "num_workers", ""),
        "seed": getattr(args, "seed", ""),
        "augmentation_profile": getattr(args, "augmentation_profile", ""),
        "foreground_crop_prob": getattr(args, "foreground_crop_prob", ""),
        "foreground_crop_min_scale": getattr(args, "foreground_crop_min_scale", ""),
        "foreground_crop_max_scale": getattr(args, "foreground_crop_max_scale", ""),
        "aux_negative_root": getattr(args, "aux_negative_root", ""),
        "aux_negative_split": getattr(args, "aux_negative_split", ""),
        "aux_negative_repeat": getattr(args, "aux_negative_repeat", ""),
        "aux_negative_augmentation_profile": getattr(args, "aux_negative_augmentation_profile", ""),
        "loss_name": getattr(args, "loss_name", ""),
        "pos_weight": getattr(args, "pos_weight", ""),
        "tversky_alpha": getattr(args, "tversky_alpha", ""),
        "tversky_beta": getattr(args, "tversky_beta", ""),
        "tversky_gamma": getattr(args, "tversky_gamma", ""),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "init_checkpoint_path": getattr(args, "init_checkpoint_path", ""),
        "best_epoch": best_epoch if best_epoch is not None else "",
        "dataset_size": dataset_size,
        "eval_threshold": getattr(args, "eval_threshold", ""),
        "postprocess_min_area": getattr(args, "postprocess_min_area", ""),
        "postprocess_max_fill_ratio": getattr(args, "postprocess_max_fill_ratio", ""),
        "postprocess_min_aspect_ratio": getattr(args, "postprocess_min_aspect_ratio", ""),
        "postprocess_max_components": getattr(args, "postprocess_max_components", ""),
        "metric_loss": metrics.get("loss", ""),
        "metric_iou": metrics.get("iou", ""),
        "metric_f1": metrics.get("f1", ""),
        "metric_precision": metrics.get("precision", ""),
        "metric_recall": metrics.get("recall", ""),
        "command": _current_command(),
    }


def _rewrite_csv_with_current_schema(csv_file: Path):
    with csv_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDNAMES)
        writer.writeheader()
        for row in existing_rows:
            normalized = {key: row.get(key, "") for key in EXPERIMENT_FIELDNAMES}
            writer.writerow(normalized)


def append_experiment_record(csv_path: str, record: dict):
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_file.exists()
    if not write_header:
        with csv_file.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header != EXPERIMENT_FIELDNAMES:
            _rewrite_csv_with_current_schema(csv_file)

    with csv_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in EXPERIMENT_FIELDNAMES})
