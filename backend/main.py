"""FastAPI backend for the AI-Assisted Medical Imaging educational prototype.

The chest X-ray classifier and Grad-CAM endpoints are separate from the
experimental CT segmentation endpoint. This software is not for clinical use.
"""

import io
import json
import tempfile
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    ScaleIntensityRange,
)
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

from backend.gradcam import create_overlay, generate_gradcam


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "chest_xray_resnet18.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
SEGMENTATION_MODEL_PATH = MODEL_DIR / "lung_tumor_unet.pth"

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_CT_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
DEVICE = torch.device("cpu")

app = FastAPI(
    title="AI-Assisted Medical Imaging System",
    description=(
        "Educational chest X-ray classification and experimental CT "
        "segmentation prototype. Not for clinical diagnosis."
    ),
    version="0.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Chest X-ray classification model
# ---------------------------------------------------------------------------
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_model():
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not CLASS_NAMES_PATH.is_file():
        raise RuntimeError(f"Class names file not found: {CLASS_NAMES_PATH}")

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        classes = json.load(file)

    if classes != ["NORMAL", "PNEUMONIA"]:
        raise RuntimeError(f"Unexpected model classes: {classes}")

    network = models.resnet18(weights=None)
    network.fc = nn.Linear(network.fc.in_features, len(classes))
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    network.load_state_dict(state)
    network.to(DEVICE)
    network.eval()
    return network, classes


model, class_names = load_model()


def read_image(content: bytes):
    """Validate image bytes and return a detached RGB PIL image."""
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail="Upload a valid JPEG or PNG image.",
                )
            image.verify()

        with Image.open(BytesIO(content)) as image:
            return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid readable image.",
        ) from exc


async def get_image_upload(file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG or PNG image only.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    image = read_image(content)
    return content, image


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>AI Medical Imaging</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>
      <body style="font-family:Arial;max-width:700px;margin:40px auto;padding:16px">
        <h1>AI-Assisted Medical Imaging System</h1>
        <p>Educational chest X-ray classification prototype.</p>
        <p><strong>Not for clinical diagnosis or treatment decisions.</strong></p>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <label>Select a chest X-ray JPEG or PNG image:</label><br><br>
          <input type="file" name="file" accept="image/jpeg,image/png" required>
          <button type="submit">Analyze image</button>
        </form>
        <p><a href="/docs">API documentation</a></p>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "segmentation_model_loaded": SEGMENTATION_MODEL_PATH.is_file(),
        "device": str(DEVICE),
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        content, image = await get_image_upload(file)
        return {
            "message": "Image uploaded and validated successfully.",
            "filename": file.filename,
            "format": image.format or "JPEG/PNG image",
            "width": image.width,
            "height": image.height,
            "size_bytes": len(content),
            "notice": "No medical diagnosis has been performed.",
        }
    finally:
        await file.close()


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        _, image = await get_image_upload(file)
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_index = int(probabilities.argmax().item())
        predicted_class = class_names[predicted_index]
        score = float(probabilities[predicted_index].item())

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "model_score": round(score, 4),
            "class_scores": {
                name: round(float(probabilities[i].item()), 4)
                for i, name in enumerate(class_names)
            },
            "notice": (
                "Educational model output only. This score is not a calibrated "
                "probability of disease and must not guide clinical decisions."
            ),
        }
    finally:
        await file.close()


@app.post("/explain")
async def explain_image(file: UploadFile = File(...)):
    try:
        _, image = await get_image_upload(file)
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        target_layer = model.layer4[-1]

        with torch.no_grad():
            logits = model(tensor)
            predicted_index = int(logits.argmax(dim=1).item())

        cam = generate_gradcam(
            model=model,
            input_tensor=tensor,
            target_layer=target_layer,
            target_index=predicted_index,
        )
        overlay = create_overlay(image, cam)

        return Response(
            content=overlay.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": 'inline; filename="gradcam_overlay.png"',
                "X-Predicted-Class": class_names[predicted_index],
                "X-Notice": "Educational visualization only; not a clinical diagnosis.",
            },
        )
    finally:
        await file.close()


# ---------------------------------------------------------------------------
# Experimental 3D CT segmentation model
# ---------------------------------------------------------------------------
# This model was trained briefly on a tiny subset and evaluated poorly. It is
# included only to demonstrate the API workflow, not for medical use.

def load_segmentation_model():
    if not SEGMENTATION_MODEL_PATH.is_file():
        raise RuntimeError(f"3D U-Net model file is missing: {SEGMENTATION_MODEL_PATH}")

    network = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
    )
    state = torch.load(
        SEGMENTATION_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    network.load_state_dict(state)
    network.eval()
    return network


segmentation_model = load_segmentation_model()
ct_loader = LoadImage(image_only=True)
ct_preprocess = Compose(
    [
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
)


@app.post("/segment-ct")
async def segment_ct(file: UploadFile = File(...)):
    """Return an experimental NIfTI segmentation mask for a NIfTI CT upload."""
    filename = (file.filename or "").lower()
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Upload a NIfTI CT scan (.nii or .nii.gz).",
        )

    content = await file.read(MAX_CT_FILE_SIZE + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_CT_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="CT file exceeds the 100 MB upload limit.",
        )

    input_suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
    input_path = None
    output_path = None

    try:
        # Close the temporary input file before MONAI opens it (Windows-safe).
        with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as tmp:
            tmp.write(content)
            input_path = Path(tmp.name)

        image = ct_loader(str(input_path))
        image = ct_preprocess(image)
        image_tensor = torch.as_tensor(np.asarray(image), dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)  # [B, C, H, W, D]

        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=segmentation_model,
                overlap=0.25,
                sw_device="cpu",
                device="cpu",
            )
            mask = logits.argmax(dim=1).squeeze(0).to(torch.uint8)

        # Keep the affine from MONAI's transformed image metadata when available.
        affine = np.eye(4, dtype=np.float64)
        try:
            meta_affine = image.affine
            if meta_affine is not None:
                affine = meta_affine.detach().cpu().numpy()
        except (AttributeError, TypeError, RuntimeError):
            pass

        nifti_mask = nib.Nifti1Image(mask.cpu().numpy().astype(np.uint8), affine)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as out_tmp:
            output_path = Path(out_tmp.name)
        nib.save(nifti_mask, str(output_path))
        output_bytes = output_path.read_bytes()

        return Response(
            content=output_bytes,
            media_type="application/gzip",
            headers={
                "Content-Disposition": 'attachment; filename="experimental_ct_mask.nii.gz"',
                "X-Model-Status": "experimental-not-validated",
                "X-Notice": "Research demo only; not for diagnosis or treatment decisions.",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process CT scan: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        await file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
