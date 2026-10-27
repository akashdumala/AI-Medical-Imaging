
from pathlib import Path
import json
from monai.data import load_decathlon_datalist
from monai.transforms import LoadImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "segmentation" / "Task06_Lung"

print("Dataset folder:", DATA_DIR)

# Check required files and dataset metadata
json_path = DATA_DIR / "dataset.json"
if not json_path.exists():
    raise FileNotFoundError(f"Missing dataset.json: {json_path}")

with open(json_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Dataset:", metadata.get("name", "Task06_Lung"))
print("Modality:", metadata.get("modality", {}))
print("Labels:", metadata.get("labels", {}))

# Read image/label pairs from MONAI's dataset metadata
data_list = load_decathlon_datalist(
    str(json_path),
    is_segmentation=True,
    data_list_key="training",
    base_dir=str(DATA_DIR),
)

print("Image-mask pairs:", len(data_list))

if not data_list:
    raise RuntimeError("No training image-mask pairs found.")

# Load one CT scan and its matching mask
sample = data_list[0]
image_path = Path(sample["image"])
label_path = Path(sample["label"])

print("\nFirst CT:", image_path)
print("First mask:", label_path)

if not image_path.is_absolute():
    image_path = DATA_DIR / image_path
if not label_path.is_absolute():
    label_path = DATA_DIR / label_path

if not image_path.exists():
    raise FileNotFoundError(f"CT image not found: {image_path}")
if not label_path.exists():
    raise FileNotFoundError(f"Mask not found: {label_path}")

loader = LoadImage(image_only=True)
image = loader(str(image_path))
mask = loader(str(label_path))

print("\nCT shape:", image.shape)
print("Mask shape:", mask.shape)
print("CT data type:", image.dtype)
print("Mask data type:", mask.dtype)
print("Mask label values:", sorted(set(mask.flatten().tolist()))[:20])
print("\nSUCCESS: CT and segmentation mask loaded.")

# Verify original NIfTI spatial metadata
import nibabel as nib
import numpy as np

ct_nii = nib.load(str(image_path))
mask_nii = nib.load(str(label_path))

print("\n--- ORIGINAL NIFTI METADATA ---")
print("CT shape:", ct_nii.shape)
print("Mask shape:", mask_nii.shape)
print("CT orientation:", nib.aff2axcodes(ct_nii.affine))
print("Mask orientation:", nib.aff2axcodes(mask_nii.affine))
print("CT voxel spacing:", ct_nii.header.get_zooms()[:3])
print("Mask voxel spacing:", mask_nii.header.get_zooms()[:3])
print("Affines match:", np.allclose(ct_nii.affine, mask_nii.affine))
print("CT affine:\n", ct_nii.affine)
print("Mask affine:\n", mask_nii.affine)

original_mask = np.asarray(mask_nii.dataobj)
print("Original mask labels:", np.unique(original_mask))
print("Original foreground voxels:", int((original_mask == 1).sum()))
