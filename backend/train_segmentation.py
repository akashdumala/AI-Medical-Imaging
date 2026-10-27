
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.data import Dataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    EnsureTyped,
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss


# ---------------- CONFIGURATION ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "segmentation" / "Task06_Lung"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "lung_tumor_unet.pth"

SEED = 42
MAX_CASES = 20       # CPU pilot; increase after checking runtime
EPOCHS = 20
PATCH_SIZE = (96, 96, 96)
VALIDATION_INTERVAL = 2
LEARNING_RATE = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

device = torch.device("cpu")
print("Device:", device)


# ---------------- DATASET ----------------
with open(DATA_DIR / "dataset.json", encoding="utf-8") as f:
    metadata = json.load(f)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return DATA_DIR / path


pairs = [
    {
        "image": str(resolve_path(item["image"])),
        "label": str(resolve_path(item["label"])),
    }
    for item in metadata["training"]
]

# Verify paths before training.
for item in pairs:
    if not Path(item["image"]).is_file():
        raise FileNotFoundError(item["image"])
    if not Path(item["label"]).is_file():
        raise FileNotFoundError(item["label"])

random.shuffle(pairs)
pairs = pairs[:min(MAX_CASES, len(pairs))]

if len(pairs) < 5:
    raise RuntimeError(
        f"Only {len(pairs)} image-mask pairs found. "
        "Check the dataset paths."
    )

split = max(1, int(len(pairs) * 0.8))
split = min(split, len(pairs) - 1)

train_pairs = pairs[:split]
val_pairs = pairs[split:]

print(f"Total pilot cases: {len(pairs)}")
print(f"Training cases: {len(train_pairs)}")
print(f"Validation cases: {len(val_pairs)}")
print("Note: validation below uses patches, not full CT volumes.")


# ---------------- TRANSFORMS ----------------
common_transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
   Orientationd(
    keys=["image", "label"],
    axcodes="RAS",
),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,
        a_max=1000,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
]

train_transforms = Compose(
    common_transforms
    + [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    common_transforms
    + [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)

train_dataset = Dataset(train_pairs, transform=train_transforms)
val_dataset = Dataset(val_pairs, transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=list_data_collate,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=list_data_collate,
)


# ---------------- MODEL ----------------
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
    num_res_units=1,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def calculate_foreground_dice(logits, labels):
    predictions = logits.argmax(dim=1)
    targets = labels.squeeze(1).long()

    predictions = predictions == 1
    targets = targets == 1

    intersection = (predictions & targets).sum().item()
    total = predictions.sum().item() + targets.sum().item()

    # Both empty is not counted as perfect tumor segmentation.
    if total == 0:
        return float("nan")

    return (2.0 * intersection) / total


# ---------------- TRAINING ----------------
best_val_dice = -1.0

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_steps = 0

    for batch in train_loader:
        images = batch["image"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_steps += 1

    mean_train_loss = train_loss_sum / max(train_steps, 1)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} "
        f"- train loss: {mean_train_loss:.4f}"
    )

    if (epoch + 1) % VALIDATION_INTERVAL != 0 and epoch + 1 != EPOCHS:
        continue

    model.eval()
    val_loss_sum = 0.0
    val_steps = 0
    dice_scores = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, dtype=torch.float32)
            labels = batch["label"].to(device).long()

            logits = model(images)
            loss = loss_function(logits, labels)

            val_loss_sum += loss.item()
            val_steps += 1

            dice = calculate_foreground_dice(logits, labels)
            if np.isfinite(dice):
                dice_scores.append(dice)

    mean_val_loss = val_loss_sum / max(val_steps, 1)
    mean_val_dice = (
        float(np.mean(dice_scores)) if dice_scores else 0.0
    )

    print(f"Validation loss: {mean_val_loss:.4f}")
    print(f"Validation patch Dice: {mean_val_dice:.4f}")

    if mean_val_dice > best_val_dice:
        best_val_dice = mean_val_dice
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved best checkpoint: {MODEL_PATH}")

print("\nTraining run finished.")
print(f"Best validation patch Dice: {best_val_dice:.4f}")
print("Important: patch Dice is not full-volume evaluation.")
print("This pilot model is not validated for clinical use.")
