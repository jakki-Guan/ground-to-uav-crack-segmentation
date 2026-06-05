import argparse
import os

import torch
from torch.utils.data import DataLoader

from crack_detection.dataset import CrackDataset
from crack_detection.experiment_logger import append_experiment_record, build_experiment_record
from crack_detection.loss import build_loss
from crack_detection.metrics import confusion_stats
from crack_detection.model import SEGFORMER_B2_MODEL_NAME, default_checkpoint_path, get_model
from crack_detection.postprocess import build_postprocess_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a crack segmentation checkpoint.")
    parser.add_argument("--dataset-root", default="CRACK500")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-name", default="Unet")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--pretrained-model-name", default=SEGFORMER_B2_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--results-csv", default="results/experiments.csv")
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
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--postprocess-min-area", type=int, default=0)
    parser.add_argument("--postprocess-max-fill-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-min-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--postprocess-max-components", type=int, default=0)
    return parser.parse_args()


def evaluate(model, loader, criterion, device, dataset_size, eval_threshold=0.5, postprocess_config=None):
    model.eval()

    test_loss = 0.0
    test_iou = 0.0
    test_f1 = 0.0
    test_precision = 0.0
    test_recall = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            batch_size_now = images.size(0)
            test_loss += loss.item() * batch_size_now
            batch_metrics = confusion_stats(
                outputs,
                masks,
                threshold=eval_threshold,
                postprocess_config=postprocess_config,
            )
            test_iou += batch_metrics["iou"] * batch_size_now
            test_f1 += batch_metrics["f1"] * batch_size_now
            test_precision += batch_metrics["precision"] * batch_size_now
            test_recall += batch_metrics["recall"] * batch_size_now

    test_loss /= dataset_size
    test_iou /= dataset_size
    test_f1 /= dataset_size
    test_precision /= dataset_size
    test_recall /= dataset_size

    return test_loss, test_iou, test_f1, test_precision, test_recall


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint_path or default_checkpoint_path(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
    )

    dataset = CrackDataset(
        root=args.dataset_root,
        split=args.split,
        img_size=args.img_size,
    )
    loader = DataLoader(
        dataset,
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
    postprocess_config = build_postprocess_config(
        min_area=args.postprocess_min_area,
        max_fill_ratio=args.postprocess_max_fill_ratio,
        min_aspect_ratio=args.postprocess_min_aspect_ratio,
        max_components=args.postprocess_max_components,
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}. "
            "Run training first or pass --checkpoint-path explicitly."
        )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Evaluation split: {args.split}")
    print(f"Samples: {len(dataset)}")
    if args.experiment_name:
        print(f"Experiment name: {args.experiment_name}")
    print(f"Loss used for reporting: {args.loss_name}")
    print(f"Eval threshold: {args.eval_threshold}")
    if postprocess_config is not None:
        print(
            "Eval postprocess: "
            f"min_area={postprocess_config.min_area}, "
            f"max_fill_ratio={postprocess_config.max_fill_ratio}, "
            f"min_aspect_ratio={postprocess_config.min_aspect_ratio}, "
            f"max_components={postprocess_config.max_components}"
        )

    test_loss, test_iou, test_f1, test_precision, test_recall = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        dataset_size=len(dataset),
        eval_threshold=args.eval_threshold,
        postprocess_config=postprocess_config,
    )

    print(
        f"Test | loss={test_loss:.4f} | "
        f"iou={test_iou:.4f} | "
        f"f1={test_f1:.4f} | "
        f"precision={test_precision:.4f} | "
        f"recall={test_recall:.4f}"
    )

    record = build_experiment_record(
        args=args,
        script_name="test.py",
        stage="test",
        split=args.split,
        checkpoint_path=checkpoint_path,
        dataset_size=len(dataset),
        metrics={
            "loss": test_loss,
            "iou": test_iou,
            "f1": test_f1,
            "precision": test_precision,
            "recall": test_recall,
        },
    )
    append_experiment_record(args.results_csv, record)
    print(f"Logged evaluation summary to {args.results_csv}")


if __name__ == "__main__":
    main()
