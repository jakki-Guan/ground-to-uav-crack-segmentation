import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crack_detection.dataset import CrackDataset
from crack_detection.model import SEGFORMER_B2_MODEL_NAME, get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit a single temperature-scaling parameter on a held-out split so that "
            "binary crack probabilities are better calibrated before downstream use."
        )
    )
    parser.add_argument("--dataset-root", default="UAV_Crack_Segmentation_Kaggle")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model-name", default="segformer-b2")
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--pretrained-model-name", default=SEGFORMER_B2_MODEL_NAME)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--img-size", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Output JSON path. Default: generated/temperature_scaling/"
            "<checkpoint_stem>_<split>.json"
        ),
    )
    return parser.parse_args()


def default_output_json(checkpoint_path: str, split: str) -> Path:
    checkpoint_stem = Path(checkpoint_path).stem
    return Path("generated/temperature_scaling") / f"{checkpoint_stem}_{split}.json"


def collect_logits_and_targets(model, loader: DataLoader, device: torch.device):
    logits_batches = []
    target_batches = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images).detach().cpu().reshape(-1).float()
            targets = masks.unsqueeze(1).detach().cpu().reshape(-1).float()
            logits_batches.append(logits)
            target_batches.append(targets)
    return torch.cat(logits_batches), torch.cat(target_batches)


def binary_ece(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: torch.Tensor,
    num_bins: int = 15,
) -> float:
    probs = torch.sigmoid(logits / temperature)
    preds = (probs >= 0.5).float()
    confidences = torch.where(preds > 0.5, probs, 1.0 - probs)
    accuracies = (preds == targets).float()

    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for bin_idx in range(num_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]
        if bin_idx == num_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        if not in_bin.any():
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = accuracies[in_bin].mean()
        ece = ece + in_bin.float().mean() * torch.abs(bin_accuracy - bin_confidence)
    return float(ece.item())


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int,
    lr: float,
) -> float:
    log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_json = Path(args.output_json) if args.output_json else default_output_json(
        args.checkpoint_path,
        args.split,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    dataset = CrackDataset(args.dataset_root, split=args.split, img_size=args.img_size)
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
    state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Calibration split: {args.split} ({len(dataset)} samples)")
    print(f"Checkpoint: {args.checkpoint_path}")

    logits_cpu, targets_cpu = collect_logits_and_targets(model, loader, device)
    logits = logits_cpu.to(device)
    targets = targets_cpu.to(device)

    initial_temperature = torch.tensor(1.0, device=device)
    initial_bce = float(F.binary_cross_entropy_with_logits(logits, targets).item())
    initial_ece = binary_ece(
        logits=logits,
        targets=targets,
        temperature=initial_temperature,
        num_bins=args.ece_bins,
    )

    fitted_temperature = fit_temperature(
        logits=logits,
        targets=targets,
        max_iter=args.max_iter,
        lr=args.lr,
    )

    fitted_temperature_tensor = torch.tensor(fitted_temperature, device=device)
    calibrated_bce = float(
        F.binary_cross_entropy_with_logits(logits / fitted_temperature_tensor, targets).item()
    )
    calibrated_ece = binary_ece(
        logits=logits,
        targets=targets,
        temperature=fitted_temperature_tensor,
        num_bins=args.ece_bins,
    )

    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": args.split,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "model_name": args.model_name,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_samples": len(dataset),
        "num_pixels": int(targets.numel()),
        "positive_pixel_ratio": float(targets.float().mean().item()),
        "optimization": {
            "method": "lbfgs",
            "max_iter": args.max_iter,
            "lr": args.lr,
        },
        "temperature_definition": "calibrated_prob = sigmoid(logits / temperature)",
        "initial_temperature": 1.0,
        "fitted_temperature": fitted_temperature,
        "metrics": {
            "bce_before": initial_bce,
            "bce_after": calibrated_bce,
            "ece_before": initial_ece,
            "ece_after": calibrated_ece,
            "ece_bins": args.ece_bins,
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Initial BCE: {initial_bce:.6f}")
    print(f"Calibrated BCE: {calibrated_bce:.6f}")
    print(f"Initial ECE: {initial_ece:.6f}")
    print(f"Calibrated ECE: {calibrated_ece:.6f}")
    print(f"Fitted temperature: {fitted_temperature:.6f}")
    print(f"Saved temperature summary to: {output_json}")


if __name__ == "__main__":
    main()
