
from pathlib import Path
import torch
from monai.networks.nets import UNet

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lung_tumor_unet.pth"

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
    num_res_units=1,
)

state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

# Verify that the saved model accepts a 3D input patch
sample = torch.randn(1, 1, 96, 96, 96)

with torch.no_grad():
    output = model(sample)

print("Model loaded successfully.")
print("Input shape:", tuple(sample.shape))
print("Output shape:", tuple(output.shape))
print("Model file size (MB):", round(MODEL_PATH.stat().st_size / (1024 * 1024), 2))
print("SUCCESS: 3D U-Net inference test passed.")
