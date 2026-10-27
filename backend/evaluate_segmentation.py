
from pathlib import Path
import json
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    Orientation, ScaleIntensityRange
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "segmentation" / "Task06_Lung"
MODEL_PATH = ROOT / "models" / "lung_tumor_unet.pth"

with open(DATA_DIR / "dataset.json", encoding="utf-8") as f:
    metadata = json.load(f)

# Evaluate on the first listed case, using its matching label.
item = metadata["training"][0]

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else DATA_DIR / p

image_path = resolve(item["image"])
label_path = resolve(item["label"])

preprocess_image = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
])

preprocess_label = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
])

image = preprocess_image(str(image_path))
label = preprocess_label(str(label_path))
# Diagnostic checks before model inference
print("\n--- DATASET DIAGNOSTICS ---")
print("Image:", image_path.name)
print("Label:", label_path.name)
print("Image shape:", tuple(image.shape))
print("Label shape:", tuple(label.shape))

print("Image intensity range:",
      float(image.min()), float(image.max()))

print("Label values:", torch.unique(
    torch.as_tensor(np.asarray(label))
).tolist())

print("Ground-truth foreground voxels:",
      int((torch.as_tensor(np.asarray(label)) == 1).sum()))

print("Shapes match:", tuple(image.shape) == tuple(label.shape))

image = torch.as_tensor(np.asarray(image), dtype=torch.float32).unsqueeze(0)
label = torch.as_tensor(np.asarray(label), dtype=torch.float32).squeeze().long()

model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(8, 16, 32, 64), strides=(2, 2, 2), num_res_units=1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

print("Evaluating:", image_path.name)

with torch.no_grad():
    logits = sliding_window_inference(
        image,
        (96, 96, 96),
        1,
        model,
        overlap=0.25,
        sw_device="cpu",
        device="cpu",
    )

    probabilities = torch.softmax(logits, dim=1)
    foreground_probability = probabilities[:, 1]

    prediction = logits.argmax(dim=1).squeeze(0)

    print(
        "Foreground probability min / mean / max:",
        float(foreground_probability.min()),
        float(foreground_probability.mean()),
        float(foreground_probability.max()),
    )

    print(
        "Foreground probability percentiles:",
        torch.quantile(
            foreground_probability.flatten(),
            torch.tensor([0.50, 0.90, 0.95, 0.99]),
        ).tolist(),
    )


truth = label == 1
pred = prediction == 1

intersection = torch.logical_and(truth, pred).sum().item()
denominator = truth.sum().item() + pred.sum().item()
dice = (2 * intersection / denominator) if denominator else 1.0

print("Ground-truth positive voxels:", truth.sum().item())
print("Predicted positive voxels:", pred.sum().item())
print(f"Dice score: {dice:.4f}")
print("Note: one-case evaluation is preliminary and not a clinical performance estimate.")
