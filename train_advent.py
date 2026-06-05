import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CrackDataset
from experiment_logger import append_experiment_record, build_experiment_record
from loss import build_loss
from metrics import confusion_stats
from model import SEGFORMER_B2_MODEL_NAME, canonical_model_name, get_model
from postprocess import build_postprocess_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train an ADVENT-style unsupervised domain adaptation baseline for "
            "binary crack segmentation."
        )
    )
    parser.add_argument("--dataset-root", default="CRACK500")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--target-dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--target-train-split", default="train")
    parser.add_argument("--val-dataset-root", default=None)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--model-name", default="DeepLabV3Plus")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--pretrained-model-name", default=SEGFORMER_B2_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc-lr", type=float, default=1e-4)
    parser.add_argument("--lambda-adv-target", type=float, default=1e-3)
    parser.add_argument("--lambda-ent-target", type=float, default=0.0)
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
    parser.add_argument(
        "--loss-name",
        choices=["bce_dice", "tversky", "focal_tversky"],
        default="bce_dice",
    )
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--tversky-gamma", type=float, default=1.33)
    parser.add_argument("--checkpoint-path", default="checkpoints/advent_deeplabv3plus_best.pth")
    parser.add_argument("--discriminator-checkpoint-path", default=None)
    parser.add_argument("--init-checkpoint-path", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--postprocess-min-area", type=int, default=0)
    parser.add_argument("--postprocess-max-fill-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-min-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-max-components", type=int, default=0)
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="Optional smoke-test/debug limit. Use 0 for a full epoch.",
    )
    return parser.parse_args()


class EntropyDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def binary_entropy_map(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    entropy = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
    return entropy / math.log(2.0)


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def build_target_train_dataset(args):
    return CrackDataset(
        root=args.target_dataset_root,
        split=args.target_train_split,
        img_size=args.img_size,
        augmentation_profile=args.augmentation_profile,
        foreground_crop_prob=0.0,
    )


def build_val_dataset(args):
    val_dataset_root = args.val_dataset_root or args.target_dataset_root
    return CrackDataset(
        root=val_dataset_root,
        split=args.val_split,
        img_size=args.img_size,
    )


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


def load_init_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


def normalize_encoder_weights(encoder_weights: str | None):
    if encoder_weights is None:
        return None
    if encoder_weights.strip().lower() in {"none", "null", ""}:
        return None
    return encoder_weights


def default_discriminator_checkpoint_path(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    return str(path.with_name(f"{path.stem}_discriminator{path.suffix}"))


def main():
    args = parse_args()
    if args.seed is not None:
        set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    discriminator_checkpoint_path = (
        args.discriminator_checkpoint_path
        or default_discriminator_checkpoint_path(args.checkpoint_path)
    )

    source_train_dataset = build_source_train_dataset(args)
    target_train_dataset = build_target_train_dataset(args)
    val_dataset = build_val_dataset(args)

    drop_last_train = args.batch_size > 1
    train_loader_generator = None
    if args.seed is not None:
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(args.seed)

    source_loader = DataLoader(
        source_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=drop_last_train,
        generator=train_loader_generator,
    )
    target_loader = DataLoader(
        target_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=drop_last_train,
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
        encoder_weights=normalize_encoder_weights(args.encoder_weights),
        in_channels=3,
        classes=1,
        pretrained_model_name=args.pretrained_model_name,
    ).to(device)
    load_init_checkpoint(model, args.init_checkpoint_path, device)

    discriminator = EntropyDiscriminator().to(device)

    criterion = build_loss(
        loss_name=args.loss_name,
        pos_weight=args.pos_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_gamma=args.tversky_gamma,
    )
    adversarial_criterion = nn.BCEWithLogitsLoss()
    model_optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    discriminator_optimizer = optim.AdamW(discriminator.parameters(), lr=args.disc_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=args.epochs)
    postprocess_config = build_postprocess_config(
        min_area=args.postprocess_min_area,
        max_fill_ratio=args.postprocess_max_fill_ratio,
        min_aspect_ratio=args.postprocess_min_aspect_ratio,
        max_components=args.postprocess_max_components,
    )

    print(f"Using device: {device}")
    print(f"Source train: {args.dataset_root}/{args.train_split} ({len(source_train_dataset)} samples)")
    print(
        f"Target train images: {args.target_dataset_root}/{args.target_train_split} "
        f"({len(target_train_dataset)} samples; masks ignored during ADVENT training)"
    )
    print(f"Validation: {args.val_dataset_root or args.target_dataset_root}/{args.val_split} ({len(val_dataset)} samples)")
    print(f"Model: {canonical_model_name(args.model_name)} / encoder={args.encoder_name}")
    print(f"Checkpoint path: {args.checkpoint_path}")

    best_val_iou = -1.0
    best_val_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    target_iter = iter(target_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        discriminator.train()

        running_total_loss = 0.0
        running_seg_loss = 0.0
        running_adv_loss = 0.0
        running_ent_loss = 0.0
        running_disc_loss = 0.0
        steps = 0

        for source_images, source_masks in source_loader:
            try:
                target_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images, _ = next(target_iter)
            source_images = source_images.to(device)
            source_masks = source_masks.unsqueeze(1).float().to(device)
            target_images = target_images.to(device)

            set_requires_grad(discriminator, False)
            model_optimizer.zero_grad()

            source_logits = model(source_images)
            target_logits = model(target_images)
            seg_loss = criterion(source_logits, source_masks)

            target_entropy = binary_entropy_map(target_logits)
            target_disc_logits = discriminator(target_entropy)
            source_domain_labels = torch.zeros_like(target_disc_logits)
            adv_loss = adversarial_criterion(target_disc_logits, source_domain_labels)
            ent_loss = target_entropy.mean()
            total_loss = (
                seg_loss
                + args.lambda_adv_target * adv_loss
                + args.lambda_ent_target * ent_loss
            )
            total_loss.backward()
            model_optimizer.step()

            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad()

            source_entropy = binary_entropy_map(source_logits.detach())
            target_entropy = binary_entropy_map(target_logits.detach())
            source_disc_logits = discriminator(source_entropy)
            target_disc_logits = discriminator(target_entropy)
            source_labels = torch.zeros_like(source_disc_logits)
            target_labels = torch.ones_like(target_disc_logits)
            disc_source_loss = adversarial_criterion(source_disc_logits, source_labels)
            disc_target_loss = adversarial_criterion(target_disc_logits, target_labels)
            disc_loss = 0.5 * (disc_source_loss + disc_target_loss)
            disc_loss.backward()
            discriminator_optimizer.step()

            running_total_loss += total_loss.item()
            running_seg_loss += seg_loss.item()
            running_adv_loss += adv_loss.item()
            running_ent_loss += ent_loss.item()
            running_disc_loss += disc_loss.item()
            steps += 1

            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break

        scheduler.step()
        denom = max(steps, 1)
        train_total_loss = running_total_loss / denom
        train_seg_loss = running_seg_loss / denom
        train_adv_loss = running_adv_loss / denom
        train_ent_loss = running_ent_loss / denom
        train_disc_loss = running_disc_loss / denom

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
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_total={train_total_loss:.4f} | "
            f"seg={train_seg_loss:.4f} | "
            f"adv={train_adv_loss:.4f} | "
            f"ent={train_ent_loss:.4f} | "
            f"disc={train_disc_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_precision={val_precision:.4f} | "
            f"val_recall={val_recall:.4f}"
        )

        if val_iou > best_val_iou + args.min_delta:
            best_val_iou = val_iou
            best_val_metrics = {
                "loss": val_loss,
                "iou": val_iou,
                "f1": val_f1,
                "precision": val_precision,
                "recall": val_recall,
            }
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint_path)
            torch.save(discriminator.state_dict(), discriminator_checkpoint_path)
            print(
                f"  Saved best model to {args.checkpoint_path} "
                f"and discriminator to {discriminator_checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    "Early stopping triggered: "
                    f"no val IoU improvement for {args.early_stopping_patience} epochs."
                )
                break

    print(f"Best val IoU: {best_val_iou:.4f} at epoch {best_epoch}")
    final_metrics = best_val_metrics or {
        "loss": val_loss,
        "iou": val_iou,
        "f1": val_f1,
        "precision": val_precision,
        "recall": val_recall,
    }

    record = build_experiment_record(
        args=args,
        script_name="train_advent.py",
        stage="train",
        split=args.val_split,
        checkpoint_path=args.checkpoint_path,
        dataset_size=len(val_dataset),
        metrics=final_metrics,
        best_epoch=best_epoch,
    )
    append_experiment_record(args.results_csv, record)
    print(f"Logged training summary to {args.results_csv}")


if __name__ == "__main__":
    main()
