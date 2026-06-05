import argparse
import copy
import csv
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run validation-threshold selection and frozen-threshold test "
            "reporting for a DAFormer/HRDA MMSeg checkpoint."
        )
    )
    parser.add_argument("--daformer-root", default="external/DAFormer")
    parser.add_argument("--config", default="configs/daformer/crack500_to_uav_daformer_mitb5_s0.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-split-name", default="val")
    parser.add_argument("--test-split-name", default="test")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--selection-metric", choices=["iou", "f1"], default="iou")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name", default="daformer_mitb5_crack500_to_uav_test_thrval")
    parser.add_argument("--report-title", default="DAFormer Threshold Selection Report")
    parser.add_argument("--model-name", default="DAFormer")
    parser.add_argument("--encoder-name", default="MiT-B5")
    parser.add_argument("--pretrained-model-name", default="external/DAFormer/pretrained/mit_b5.pth")
    parser.add_argument("--dataset-root", default="generated/daformer/crack500_to_uav")
    parser.add_argument("--results-csv", default="results/experiments.csv")
    parser.add_argument("--no-log-test-result", action="store_true")
    return parser.parse_args()


def parse_threshold_list(raw):
    thresholds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            thresholds.append(float(part))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return sorted(set(thresholds))


def add_daformer_to_path(daformer_root):
    root = str(Path(daformer_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def import_daformer_modules():
    import mmcv
    from mmcv.parallel import MMDataParallel
    from mmseg.datasets import build_dataloader, build_dataset
    from mmseg.models import build_segmentor

    return mmcv, MMDataParallel, build_dataloader, build_dataset, build_segmentor


def build_split_loader(cfg, split_key, num_workers, build_dataloader, build_dataset):
    data_cfg = copy.deepcopy(cfg.data[split_key])
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=num_workers,
        dist=False,
        shuffle=False,
    )
    return dataset, loader


def infer_img_size(cfg):
    crop_size = cfg.get("crop_size")
    if crop_size is not None and len(crop_size) == 2 and crop_size[0] == crop_size[1]:
        return int(crop_size[0])

    data_cfg = cfg.get("data", {})
    for split_key in ("test", "val"):
        split_cfg = data_cfg.get(split_key)
        if split_cfg is None:
            continue
        pipeline = split_cfg.get("pipeline", [])
        for step in pipeline:
            if step.get("type") != "MultiScaleFlipAug":
                continue
            img_scale = step.get("img_scale")
            if (
                img_scale is not None
                and len(img_scale) == 2
                and img_scale[0] == img_scale[1]
            ):
                return int(img_scale[0])
    return ""


def load_student_state_dict(model, checkpoint_path):
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


def init_model(cfg, checkpoint_path, device, build_segmentor, MMDataParallel):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_student_state_dict(model, checkpoint_path)
    model = model.to(device)
    if device.type == "cuda":
        model = MMDataParallel(model, device_ids=[device.index or 0])
    model.eval()
    return model


def confusion_from_prediction(probability, target, threshold):
    pred = probability >= threshold
    gt = target > 0
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.float64)
    return tp, fp, fn


def metrics_from_confusion(tp, fp, fn, eps=1e-6):
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_split(model, loader, dataset, thresholds, device):
    rows = []
    totals = {threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for threshold in thresholds}
    gt_maps = dataset.get_gt_seg_maps(efficient_test=False)

    with torch.no_grad():
        for index, data in enumerate(loader):
            img = data["img"][0].to(device)
            img_metas = data["img_metas"][0].data[0]
            segmentor = model.module if hasattr(model, "module") else model
            probabilities = segmentor.inference(img, img_metas, rescale=True)
            crack_probability = probabilities[:, 1, :, :].squeeze(0).detach().cpu().numpy()
            target = gt_maps[index]
            if crack_probability.shape != target.shape:
                raise ValueError(
                    "Prediction/target shape mismatch for sample {}: {} vs {}".format(
                        index, crack_probability.shape, target.shape
                    )
                )
            for threshold in thresholds:
                tp, fp, fn = confusion_from_prediction(crack_probability, target, threshold)
                totals[threshold]["tp"] += tp
                totals[threshold]["fp"] += fp
                totals[threshold]["fn"] += fn

    for threshold in thresholds:
        stats = totals[threshold]
        rows.append(
            {
                "threshold": threshold,
                **metrics_from_confusion(stats["tp"], stats["fp"], stats["fn"]),
            }
        )
    return rows


def write_metric_csv(path, split, rows):
    fieldnames = ["split", "threshold", "iou", "f1", "precision", "recall"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"split": split, **row})


def write_summary(path, args, val_size, test_size, selected_row, test_row):
    lines = [
        "# {}".format(args.report_title),
        "",
        "- Config: `{}`".format(args.config),
        "- Checkpoint: `{}`".format(args.checkpoint),
        "- Validation split: `{}` ({} samples)".format(args.val_split_name, val_size),
        "- Test split: `{}` ({} samples)".format(args.test_split_name, test_size),
        "- Selection metric: `{}`".format(args.selection_metric),
        "- Selected threshold: `{:.2f}`".format(selected_row["threshold"]),
        "",
        "## Validation Selection",
        "",
        "- IoU `{iou:.4f}`, F1 `{f1:.4f}`, precision `{precision:.4f}`, recall `{recall:.4f}`".format(
            **selected_row
        ),
        "",
        "## Confirmatory Test",
        "",
        "- IoU `{iou:.4f}`, F1 `{f1:.4f}`, precision `{precision:.4f}`, recall `{recall:.4f}`".format(
            **test_row
        ),
        "",
        "Threshold selection used validation data only. The test split was evaluated once with the frozen selected threshold.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_experiment_record(args, cfg, selected_threshold, test_row, test_size):
    csv_path = Path(args.results_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "timestamp_utc": timestamp,
        "run_id": "{}_{}".format(timestamp, os.getpid()),
        "script": "scripts/daformer/evaluate_daformer_threshold_sweep.py",
        "stage": "test",
        "split": args.test_split_name,
        "experiment_name": args.experiment_name,
        "dataset_root": args.dataset_root,
        "train_split": "source=train,target=train",
        "val_dataset_root": "{}/target".format(args.dataset_root),
        "val_split": args.val_split_name,
        "model_name": args.model_name,
        "encoder_name": args.encoder_name,
        "encoder_weights": "imagenet",
        "pretrained_model_name": args.pretrained_model_name,
        "img_size": infer_img_size(cfg),
        "batch_size": 1,
        "num_workers": args.num_workers,
        "seed": 42,
        "loss_name": "cross_entropy",
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "dataset_size": test_size,
        "eval_threshold": selected_threshold,
        "metric_iou": test_row["iou"],
        "metric_f1": test_row["f1"],
        "metric_precision": test_row["precision"],
        "metric_recall": test_row["recall"],
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
    }

    write_header = not csv_path.exists()
    if not write_header:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header and existing_header != EXPERIMENT_FIELDNAMES:
            raise ValueError(
                "Unexpected results CSV schema in {}. Refusing to append DAFormer row.".format(
                    csv_path
                )
            )

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in EXPERIMENT_FIELDNAMES})


def main():
    args = parse_args()
    thresholds = parse_threshold_list(args.thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    add_daformer_to_path(args.daformer_root)
    mmcv, MMDataParallel, build_dataloader, build_dataset, build_segmentor = import_daformer_modules()
    cfg = mmcv.Config.fromfile(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    val_dataset, val_loader = build_split_loader(
        cfg, "val", args.num_workers, build_dataloader, build_dataset
    )
    test_dataset, test_loader = build_split_loader(
        cfg, "test", args.num_workers, build_dataloader, build_dataset
    )
    model = init_model(
        cfg, args.checkpoint, device, build_segmentor, MMDataParallel
    )

    print("Using device: {}".format(device))
    print("Validation samples: {}".format(len(val_dataset)))
    print("Test samples: {}".format(len(test_dataset)))
    print("Thresholds: {}".format(", ".join("{:.2f}".format(t) for t in thresholds)))

    val_rows = evaluate_split(model, val_loader, val_dataset, thresholds, device)
    for row in val_rows:
        print(
            "Val threshold={threshold:.2f} | iou={iou:.4f} | f1={f1:.4f} | "
            "precision={precision:.4f} | recall={recall:.4f}".format(**row)
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
    selected_threshold = selected_row["threshold"]
    test_rows = evaluate_split(model, test_loader, test_dataset, [selected_threshold], device)
    test_row = test_rows[0]

    write_metric_csv(output_dir / "threshold_sweep_val.csv", args.val_split_name, val_rows)
    write_metric_csv(output_dir / "selected_threshold_test.csv", args.test_split_name, [test_row])
    selection = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selection_metric": args.selection_metric,
        "threshold_candidates": thresholds,
        "selected_threshold": selected_threshold,
        "selected_val_metrics": selected_row,
        "test_metrics": test_row,
        "metric_definition": "foreground crack metrics from class-1 softmax probability thresholding",
    }
    (output_dir / "selected_threshold.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    write_summary(
        output_dir / "summary.md",
        args,
        len(val_dataset),
        len(test_dataset),
        selected_row,
        test_row,
    )

    print(
        "Selected threshold={:.2f} on val ({}={:.4f})".format(
            selected_threshold, args.selection_metric, selected_row[args.selection_metric]
        )
    )
    print(
        "Test @ {:.2f} | iou={:.4f} | f1={:.4f} | precision={:.4f} | recall={:.4f}".format(
            selected_threshold,
            test_row["iou"],
            test_row["f1"],
            test_row["precision"],
            test_row["recall"],
        )
    )
    print("Wrote report assets to {}".format(output_dir))

    if not args.no_log_test_result:
        append_experiment_record(args, cfg, selected_threshold, test_row, len(test_dataset))
        print("Logged selected-threshold test result to {}".format(args.results_csv))


if __name__ == "__main__":
    main()
