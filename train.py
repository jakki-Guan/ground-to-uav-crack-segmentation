import os
import sys 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CrackDataset
from model import get_model 
from metrics import iou_score, f1_score
from loss import BCEDiceLoss

epochs = 30  # A solid starting point: enough epochs to learn, but still short enough for quick iteration.
batch_size = 16  # Reasonable for a 12GB GPU; larger batches are stabler but use more VRAM.
lr = 1e-4  # Conservative AdamW learning rate that usually trains segmentation models without blowing up.
img_size = 360  # Matches the dataset transform default and keeps memory use manageable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU when available, otherwise fall back to CPU.
checkpoint_dir = "checkpoints"  # Save trained model weights here so the best run is easy to reload later.
early_stopping_patience = 5  # Stop if validation IoU does not meaningfully improve for 5 epochs in a row.
min_delta = 1e-3  # Tiny IoU gains smaller than this are treated as noise, not real improvement.

os.makedirs(checkpoint_dir, exist_ok=True)

train_dataset = CrackDataset(root="CRACK500", split="train", img_size=img_size)  # Training split: used to update model weights.
val_dataset   = CrackDataset(root="CRACK500", split="val", img_size=img_size)  # Validation split: used to judge generalization after each epoch.


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)  # Shuffle for better SGD mixing; 0 workers is the simplest/debug-friendly setting.
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)  # No shuffle for evaluation consistency; 0 workers avoids multiprocessing quirks at first.

model=get_model(model_name="Unet", encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)  # U-Net is a reliable baseline; ResNet34 + ImageNet weights gives a strong, lightweight starting encoder.
model = model.to(device)
criterion = BCEDiceLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)  # AdamW is an easy, strong default optimizer for modern vision training.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Gradually lowers the learning rate across the whole training run.

best_iou = 0.0  # Track the best validation IoU so we only save the strongest checkpoint.
best_epoch = 0  # Record which epoch produced the best validation IoU.
epochs_without_improvement = 0  # Count how long validation IoU has gone without a meaningful gain.

for epoch in range(epochs):

    # --- train ---
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.unsqueeze(1).float().to(device)
        optimizer.zero_grad()  # Clear gradients from the previous step.
        outputs = model(images)  # Forward pass: compute predicted masks.
        loss = criterion(outputs, masks)  # Compute the combined BCE + Dice loss.
        loss.backward()  # Backpropagate to compute gradients.
        optimizer.step()  # Update model weights based on computed gradients.
        train_loss += loss.item() * images.size(0)  # Accumulate total loss
        # 1. 数据移到 device
        # 2. 清零梯度
        # 3. forward
        # 4. 算 loss
        # 5. backward
        # 6. optimizer step
        # 7. 累加 loss

    # --- validate ---
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_f1 = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            # 1. 数据移到 device
            # 2. forward
            # 3. 累加 iou 和 f1
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)
            val_iou += iou_score(outputs, masks) * images.size(0)
            val_f1 += f1_score(outputs, masks) * images.size(0)

    # --- epoch 结束 ---
    # 1. 打印 epoch / train_loss / val_iou / val_f1
    # 2. scheduler.step()
    # 3. 如果 val_iou > best_iou，保存 checkpoint
    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)
    val_iou /= len(val_dataset)
    val_f1 /= len(val_dataset)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_iou={val_iou:.4f} | "
        f"val_f1={val_f1:.4f}"
    )

    scheduler.step()

    if val_iou > best_iou + min_delta:
        best_iou = val_iou
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        print(f"Saved new best checkpoint at epoch {best_epoch} with val_iou={best_iou:.4f}")
    else:
        epochs_without_improvement += 1
        print(
            f"No significant val_iou improvement for {epochs_without_improvement} epoch(s). "
            f"Best is still epoch {best_epoch} with val_iou={best_iou:.4f}"
        )

    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break
