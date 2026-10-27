
from pathlib import Path
import json
import copy
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility and CPU-friendly settings
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 3
IMAGE_SIZE = 224

print(f"Device: {DEVICE}")
print(f"Dataset: {DATA_DIR}")

# Validate dataset structure
required_dirs = [
    DATA_DIR / "train",
    DATA_DIR / "val",
    DATA_DIR / "test",
]

for directory in required_dirs:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing directory: {directory}\n"
            "Check the data/chest_xray folder structure."
        )

# Image preprocessing and training augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_data = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform,
)
val_data = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform,
)
test_data = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform,
)

class_names = train_data.classes

if class_names != ["NORMAL", "PNEUMONIA"]:
    raise ValueError(
        f"Unexpected class folders: {class_names}. "
        "Expected NORMAL and PNEUMONIA."
    )

if val_data.classes != class_names or test_data.classes != class_names:
    raise ValueError("Train, validation and test classes must match.")

for name, dataset in [
    ("Train", train_data),
    ("Validation", val_data),
    ("Test", test_data),
]:
    counts = {
        class_name: sum(
            1 for label in dataset.targets
            if label == dataset.class_to_idx[class_name]
        )
        for class_name in class_names
    }
    print(f"{name}: {len(dataset)} images; {counts}")

if min(len(train_data), len(val_data), len(test_data)) == 0:
    raise ValueError("One or more dataset splits are empty.")

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)
test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# Load pretrained ResNet-18 weights
print("Loading pretrained ResNet-18...")
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze the early layers to reduce training time.
for parameter in model.parameters():
    parameter.requires_grad = False

# Fine-tune the final feature block and classification layer.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001,
)

best_val_accuracy = -1.0
best_state = copy.deepcopy(model.state_dict())

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            tp += ((predictions == 1) & (labels == 1)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if (batch_index + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"Batch {batch_index + 1}/{len(train_loader)}"
            )

    val_metrics = evaluate(val_loader)
    train_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {total_loss / total:.4f} | "
        f"Train accuracy: {train_accuracy:.4f} | "
        f"Validation loss: {val_metrics['loss']:.4f} | "
        f"Validation accuracy: {val_metrics['accuracy']:.4f}"
    )

    if val_metrics["accuracy"] > best_val_accuracy:
        best_val_accuracy = val_metrics["accuracy"]
        best_state = copy.deepcopy(model.state_dict())

# Save the best validation checkpoint.
model.load_state_dict(best_state)

model_path = MODEL_DIR / "chest_xray_resnet18.pth"
torch.save(model.state_dict(), model_path)

with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# Evaluate the held-out test set only after model selection.
test_metrics = evaluate(test_loader)

tp = test_metrics["tp"]
tn = test_metrics["tn"]
fp = test_metrics["fp"]
fn = test_metrics["fn"]

sensitivity = tp / (tp + fn) if tp + fn else 0.0
specificity = tn / (tn + fp) if tn + fp else 0.0
precision = tp / (tp + fp) if tp + fp else 0.0

print("\nFinal held-out test results")
print(f"Test loss:       {test_metrics['loss']:.4f}")
print(f"Test accuracy:   {test_metrics['accuracy']:.4f}")
print(f"Sensitivity:     {sensitivity:.4f}")
print(f"Specificity:     {specificity:.4f}")
print(f"Precision:       {precision:.4f}")
print(f"Confusion counts: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Model saved to: {model_path}")
print(f"Classes saved to: {MODEL_DIR / 'class_names.json'}")
