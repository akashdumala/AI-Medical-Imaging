
from pathlib import Path
import json
import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    ScaleIntensityRange,
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import NibabelWriter

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "segmentation" / "Task06_Lung"
MODEL_PATH = ROOT / "models" / "lung_tumor_unet.pth"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose a CT volume from the dataset
with open(DATA_DIR / "dataset.json", encoding="utf-8") as f:
    metadata = json.load(f)

sample = metadata["training"][0]
image_path = Path(sample["image"])
if not image_path.is_absolute():
    image_path = DATA_DIR / image_path

device = torch.device("cpu")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
    num_res_units=1,
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

preprocess = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    ScaleIntensityRange(
        a_min=-1000,
        a_max=1000,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
])

print("Loading CT:", image_path)
image = preprocess(str(image_path))
if isinstance(image, np.ndarray):
    image = torch.as_tensor(image)
image = image.float().unsqueeze(0).to(device)  # [B, C, H, W, D]

print("Input volume shape:", tuple(image.shape))
print("Running sliding-window inference on CPU; this may take a while...")

with torch.no_grad():
    logits = sliding_window_inference(
        inputs=image,
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
        sw_device=device,
        device=device,
    )
    mask = torch.argmax(logits, dim=1).squeeze(0).to(torch.uint8)

print("Predicted mask shape:", tuple(mask.shape))
print("Predicted labels:", torch.unique(mask).tolist())

output_path = OUTPUT_DIR / "predicted_lung_tumor_mask.nii.gz"
writer = NibabelWriter()
writer.set_data_array(mask.cpu().numpy()[None, ...])
writer.set_metadata({"affine": np.eye(4)})
writer.write(str(output_path), verbose=False)

print("Saved predicted mask:", output_path)
print("Note: this is an unvalidated prototype prediction, not a medical diagnosis.")
