import argparse
import csv
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CrackDataset
from experiment_logger import append_experiment_record, build_experiment_record
from loss import build_loss
from metrics import f1_score, iou_score
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill experiment CSV rows for an existing U-Net baseline checkpoint."
    )
    parser.add_argument("--dataset-root", default="CRACK500")
    parser.add_argument("--results-csv", default="results/experiments.csv")
    parser.add_argument("--checkpoint-path", default="checkpoints/best_model.pth")
    parser.add_argument("--experiment-name", default="unet_baseline_360")
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--loss-name", default="bce_dice")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--best-epoch", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def evaluate(model, loader, criterion, device, dataset_size):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            batch_size_now = images.size(0)
            total_loss += loss.item() * batch_size_now
            total_iou += iou_score(outputs, masks) * batch_size_now
            total_f1 += f1_score(outputs, masks) * batch_size_now

    return (
        total_loss / dataset_size,
        total_iou / dataset_size,
        total_f1 / dataset_size,
    )


def existing_row_keys(csv_path: Path):
    if not csv_path.exists():
        return set()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            (row.get("experiment_name", ""), row.get("stage", ""), row.get("split", ""))
            for row in reader
        }


def build_logging_args(cli_args):
    return Namespace(
        dataset_root=cli_args.dataset_root,
        model_name="Unet",
        encoder_name=cli_args.encoder_name,
        encoder_weights=cli_args.encoder_weights,
        pretrained_model_name="",
        img_size=cli_args.img_size,
        batch_size=cli_args.batch_size,
        epochs=cli_args.epochs,
        lr=cli_args.lr,
        num_workers=cli_args.num_workers,
        augmentation_profile="baseline",
        foreground_crop_prob=0.0,
        foreground_crop_min_scale=0.55,
        foreground_crop_max_scale=1.0,
        loss_name=cli_args.loss_name,
        pos_weight=None,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=1.33,
        experiment_name=cli_args.experiment_name,
    )


def make_loader(dataset_root, split, img_size, batch_size, num_workers):
    dataset = CrackDataset(
        root=dataset_root,
        split=split,
        img_size=img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, loader


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    csv_path = Path(args.results_csv)
    current_keys = existing_row_keys(csv_path)
    target_keys = [
        (args.experiment_name, "val_best", "val"),
        (args.experiment_name, "test", "test"),
    ]

    if not args.force and all(key in current_keys for key in target_keys):
        print(
            "Baseline rows already exist in the CSV. "
            "Use --force if you want to append them again."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        model_name="Unet",
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    criterion = build_loss(loss_name=args.loss_name)
    logging_args = build_logging_args(args)

    for stage, split, best_epoch in [
        ("val_best", "val", args.best_epoch),
        ("test", "test", None),
    ]:
        row_key = (args.experiment_name, stage, split)
        if row_key in current_keys and not args.force:
            print(f"Skipping existing row: experiment={args.experiment_name}, stage={stage}, split={split}")
            continue

        dataset, loader = make_loader(
            dataset_root=args.dataset_root,
            split=split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        loss_value, iou_value, f1_value = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            dataset_size=len(dataset),
        )
        record = build_experiment_record(
            args=logging_args,
            script_name="backfill_unet_baseline_records.py",
            stage=stage,
            split=split,
            checkpoint_path=str(checkpoint_path),
            dataset_size=len(dataset),
            metrics={
                "loss": loss_value,
                "iou": iou_value,
                "f1": f1_value,
            },
            best_epoch=best_epoch,
        )
        append_experiment_record(str(csv_path), record)
        print(
            f"Logged {stage}/{split}: "
            f"loss={loss_value:.4f}, iou={iou_value:.4f}, f1={f1_value:.4f}"
        )


if __name__ == "__main__":
    main()
