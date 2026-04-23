import os

import torch
from torch.utils.data import DataLoader

from dataset import CrackDataset
from loss import BCEDiceLoss
from metrics import f1_score, iou_score
from model import get_model


# --- 1) Basic config: keep these aligned with training -----------------------
# These settings should match the training script, otherwise the loaded weights
# may not fit the model or the test preprocessing may be inconsistent.
batch_size = 16  # Use the same batch size as training unless memory forces you to change it.
img_size = 360  # Must match the resize used during training/validation for fair evaluation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Run on GPU if available, otherwise use CPU.
checkpoint_path = os.path.join("checkpoints", "best_model.pth")  # This is the best checkpoint saved during training.

# Model config must be identical to the model that produced best_model.pth.
model_name = "Unet"
encoder_name = "resnet34"
encoder_weights = "imagenet"
in_channels = 3
classes = 1


def evaluate(model, loader, criterion, device, dataset_size):
    # --- 5) Evaluation loop: no gradient tracking during test ----------------
    # Testing only measures performance. We do not update weights here.
    model.eval()

    test_loss = 0.0
    test_iou = 0.0
    test_f1 = 0.0

    with torch.no_grad():
        for images, masks in loader:
            # Move the batch to the same device as the model.
            images = images.to(device)

            # Add the channel dimension so masks match model output shape:
            # masks: (N, H, W) -> (N, 1, H, W)
            # Convert to float because BCE-based losses expect float targets.
            masks = masks.unsqueeze(1).float().to(device)

            # Forward pass: get predicted logits for the batch.
            outputs = model(images)

            # Compute loss and metrics on this batch.
            loss = criterion(outputs, masks)

            # Multiply by batch size so we can compute a true dataset average later.
            batch_size_now = images.size(0)
            test_loss += loss.item() * batch_size_now
            test_iou += iou_score(outputs, masks) * batch_size_now
            test_f1 += f1_score(outputs, masks) * batch_size_now

    # Convert total sums into per-sample averages over the whole test set.
    test_loss /= dataset_size
    test_iou /= dataset_size
    test_f1 /= dataset_size

    return test_loss, test_iou, test_f1


def main():
    # --- 2) Build the test dataset and dataloader ----------------------------
    # shuffle=False keeps the sample order stable, which is better for testing
    # and for later debugging or visualization.
    test_dataset = CrackDataset(root="CRACK500", split="test", img_size=img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --- 3) Rebuild the exact same model structure used in training ----------
    # The checkpoint only stores weights, not the model definition itself.
    model = get_model(
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    model = model.to(device)

    # We reuse the same loss here so test_loss is directly comparable to
    # validation loss from training.
    criterion = BCEDiceLoss()

    # --- 4) Load the saved best checkpoint -----------------------------------
    # If this file does not exist, it usually means training has not saved a
    # best model yet, or the path is different from what we expect.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}. "
            "Run training first or update the checkpoint path in test.py."
        )

    # weights_only=True is the safer choice for loading a plain state_dict.
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_dataset)}")

    # --- 6) Run evaluation and print the final test result -------------------
    test_loss, test_iou, test_f1 = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        dataset_size=len(test_dataset),
    )

    print(
        f"Test | loss={test_loss:.4f} | "
        f"iou={test_iou:.4f} | "
        f"f1={test_f1:.4f}"
    )


if __name__ == "__main__":
    main()
