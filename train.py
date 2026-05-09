import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from dataset import CrackDataset
from experiment_logger import append_experiment_record, build_experiment_record
from loss import build_loss
from metrics import confusion_stats
from model import (
    SEGFORMER_B2_MODEL_NAME,
    canonical_model_name,
    default_checkpoint_path,
    get_model,
)
from postprocess import build_postprocess_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a crack segmentation baseline, a source-plus-negative-bank variant, "
            "or a few-shot fine-tuning run from an existing checkpoint."
        )
    )
    parser.add_argument("--dataset-root", default="CRACK500")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-dataset-root", default=None)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--model-name", default="Unet")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--pretrained-model-name", default=SEGFORMER_B2_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--results-csv", default="results/experiments.csv")
    parser.add_argument(
        "--augmentation-profile",
        choices=["baseline", "mild", "strong"],
        default="baseline",
    )
    parser.add_argument("--foreground-crop-prob", type=float, default=0.0)
    parser.add_argument("--foreground-crop-min-scale", type=float, default=0.55)
    parser.add_argument("--foreground-crop-max-scale", type=float, default=1.0)
    parser.add_argument("--aux-negative-root", default=None)
    parser.add_argument("--aux-negative-split", default="train")
    parser.add_argument("--aux-negative-repeat", type=int, default=1)
    parser.add_argument(
        "--aux-negative-augmentation-profile",
        choices=["inherit", "baseline", "mild", "strong"],
        default="inherit",
    )
    parser.add_argument(
        "--loss-name",
        choices=["bce_dice", "tversky", "focal_tversky"],
        default="bce_dice",
    )
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--tversky-gamma", type=float, default=1.33)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--init-checkpoint-path", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--postprocess-min-area", type=int, default=0)
    parser.add_argument("--postprocess-max-fill-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-min-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-max-components", type=int, default=0)
    return parser.parse_args()


def build_source_train_dataset(args):
    return CrackDataset(
        root=args.dataset_root,
        split=args.train_split,
        img_size=args.img_size,
        augmentation_profile=args.augmentation_profile,
        foreground_crop_prob=args.foreground_crop_prob,
        foreground_crop_min_scale=args.foreground_crop_min_scale,
        foreground_crop_max_scale=args.foreground_crop_max_scale,
    )


def build_aux_negative_dataset(args):
    if not args.aux_negative_root:
        return None
    if args.aux_negative_repeat < 1:
        raise ValueError("--aux-negative-repeat must be >= 1 when --aux-negative-root is set.")

    aux_augmentation_profile = (
        args.augmentation_profile
        if args.aux_negative_augmentation_profile == "inherit"
        else args.aux_negative_augmentation_profile
    )
    return CrackDataset(
        root=args.aux_negative_root,
        split=args.aux_negative_split,
        img_size=args.img_size,
        augmentation_profile=aux_augmentation_profile,
        foreground_crop_prob=0.0,
        foreground_crop_min_scale=args.foreground_crop_min_scale,
        foreground_crop_max_scale=args.foreground_crop_max_scale,
    )


def build_train_dataset(args):
    source_train_dataset = build_source_train_dataset(args)
    aux_negative_dataset = build_aux_negative_dataset(args)

    if aux_negative_dataset is None:
        return source_train_dataset, source_train_dataset, None

    mixed_train_dataset = ConcatDataset(
        [source_train_dataset] + [aux_negative_dataset] * args.aux_negative_repeat
    )
    return mixed_train_dataset, source_train_dataset, aux_negative_dataset


def should_drop_last_train_batch(model_name, dataset_len, batch_size):
    if batch_size <= 1:
        return False
    if canonical_model_name(model_name) != "deeplabv3plus":
        return False
    return dataset_len % batch_size == 1


def build_val_dataset(args):
    val_dataset_root = args.val_dataset_root or args.dataset_root
    return CrackDataset(
        root=val_dataset_root,
        split=args.val_split,
        img_size=args.img_size,
    )


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device, dataset_size, eval_threshold=0.5, postprocess_config=None):
    model.eval()

    val_loss = 0.0
    val_iou = 0.0
    val_f1 = 0.0
    val_precision = 0.0
    val_recall = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            batch_size_now = images.size(0)
            val_loss += loss.item() * batch_size_now
            batch_metrics = confusion_stats(
                outputs,
                masks,
                threshold=eval_threshold,
                postprocess_config=postprocess_config,
            )
            val_iou += batch_metrics["iou"] * batch_size_now
            val_f1 += batch_metrics["f1"] * batch_size_now
            val_precision += batch_metrics["precision"] * batch_size_now
            val_recall += batch_metrics["recall"] * batch_size_now

    val_loss /= dataset_size
    val_iou /= dataset_size
    val_f1 /= dataset_size
    val_precision /= dataset_size
    val_recall /= dataset_size

    return val_loss, val_iou, val_f1, val_precision, val_recall


def main():
    args = parse_args()
    if args.seed is not None:
        set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint_path or default_checkpoint_path(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
    )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset, source_train_dataset, aux_negative_dataset = build_train_dataset(args)
    val_dataset = build_val_dataset(args)

    drop_last_train_batch = should_drop_last_train_batch(
        model_name=args.model_name,
        dataset_len=len(train_dataset),
        batch_size=args.batch_size,
    )
    train_loader_generator = None
    if args.seed is not None:
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=drop_last_train_batch,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = get_model(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
        pretrained_model_name=args.pretrained_model_name,
    )
    model = model.to(device)

    criterion = build_loss(
        loss_name=args.loss_name,
        pos_weight=args.pos_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    postprocess_config = build_postprocess_config(
        min_area=args.postprocess_min_area,
        max_fill_ratio=args.postprocess_max_fill_ratio,
        min_aspect_ratio=args.postprocess_min_aspect_ratio,
        max_components=args.postprocess_max_components,
    )

    best_iou = float("-inf")
    best_val_loss = None
    best_val_f1 = None
    best_val_precision = None
    best_val_recall = None
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Using device: {device}")
    print(f"Training model: {args.model_name}")
    print(f"Checkpoint path: {checkpoint_path}")
    if args.experiment_name:
        print(f"Experiment name: {args.experiment_name}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print(
        f"Training setup: img_size={args.img_size}, aug={args.augmentation_profile}, "
        f"fg_crop_prob={args.foreground_crop_prob:.2f}, "
        f"loss={args.loss_name}"
    )
    print(
        f"Source train dataset: root={args.dataset_root}, split={args.train_split}, "
        f"samples={len(source_train_dataset)}"
    )
    if drop_last_train_batch:
        print(
            "Training loader: drop_last=True to avoid a final batch of size 1, "
            "which breaks DeepLabV3+ BatchNorm in train mode."
        )
    print(
        f"Validation dataset: root={args.val_dataset_root or args.dataset_root}, "
        f"split={args.val_split}, samples={len(val_dataset)}"
    )
    if aux_negative_dataset is not None:
        aux_augmentation_profile = (
            args.augmentation_profile
            if args.aux_negative_augmentation_profile == "inherit"
            else args.aux_negative_augmentation_profile
        )
        print(
            "Aux negative bank: "
            f"root={args.aux_negative_root}, "
            f"split={args.aux_negative_split}, "
            f"samples={len(aux_negative_dataset)}, "
            f"repeat={args.aux_negative_repeat}, "
            f"effective_samples={len(aux_negative_dataset) * args.aux_negative_repeat}, "
            f"aug={aux_augmentation_profile}"
        )
        print(f"Mixed train dataset size: {len(train_dataset)}")
    print(f"Eval threshold: {args.eval_threshold}")
    if postprocess_config is not None:
        print(
            "Eval postprocess: "
            f"min_area={postprocess_config.min_area}, "
            f"max_fill_ratio={postprocess_config.max_fill_ratio}, "
            f"min_aspect_ratio={postprocess_config.min_aspect_ratio}, "
            f"max_components={postprocess_config.max_components}"
        )
    if args.loss_name == "bce_dice" and args.pos_weight is not None:
        print(f"BCE positive class weight: {args.pos_weight}")
    if args.loss_name in {"tversky", "focal_tversky"}:
        print(
            f"Tversky params: alpha={args.tversky_alpha}, "
            f"beta={args.tversky_beta}, gamma={args.tversky_gamma}"
        )
    if args.model_name.lower().replace("_", "").replace("-", "") == "segformerb2":
        print(f"SegFormer weights: {args.pretrained_model_name}")
        if args.batch_size > 8:
            print("Hint: SegFormer-B2 often needs a smaller batch size such as 4 or 8.")
    if args.img_size >= 512 and args.batch_size > 8:
        print("Hint: higher image resolution often needs a smaller batch size such as 4 or 8.")
    if args.init_checkpoint_path:
        if not os.path.exists(args.init_checkpoint_path):
            raise FileNotFoundError(f"Initial checkpoint not found: {args.init_checkpoint_path}")
        init_state_dict = torch.load(args.init_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(init_state_dict)
        print(f"Initialized model weights from: {args.init_checkpoint_path}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        val_loss, val_iou, val_f1, val_precision, val_recall = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            dataset_size=len(val_dataset),
            eval_threshold=args.eval_threshold,
            postprocess_config=postprocess_config,
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_precision={val_precision:.4f} | "
            f"val_recall={val_recall:.4f}"
        )

        scheduler.step()

        if val_iou > best_iou + args.min_delta:
            best_iou = val_iou
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best checkpoint at epoch {best_epoch} with val_iou={best_iou:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"No significant val_iou improvement for {epochs_without_improvement} epoch(s). "
                f"Best is still epoch {best_epoch} with val_iou={best_iou:.4f}"
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    record = build_experiment_record(
        args=args,
        script_name="train.py",
        stage="val_best",
        split=args.val_split,
        checkpoint_path=checkpoint_path,
        dataset_size=len(val_dataset),
        metrics={
            "loss": best_val_loss if best_val_loss is not None else val_loss,
            "iou": best_iou,
            "f1": best_val_f1 if best_val_f1 is not None else val_f1,
            "precision": (
                best_val_precision if best_val_precision is not None else val_precision
            ),
            "recall": best_val_recall if best_val_recall is not None else val_recall,
        },
        best_epoch=best_epoch,
    )
    append_experiment_record(args.results_csv, record)
    print(f"Logged validation summary to {args.results_csv}")


if __name__ == "__main__":
    main()
