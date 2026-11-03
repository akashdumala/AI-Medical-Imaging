# AI-Assisted Medical Imaging and Diagnosis System

An educational medical-imaging prototype that combines deep-learning models, a Python REST API, and a React interface to explore chest X-ray classification, Grad-CAM visual explanations, and experimental CT segmentation.

> **Safety notice:** This is a research and educational prototype, not a medical device. It must not be used to diagnose patients or guide treatment.

## Features

- Chest X-ray classification into `NORMAL` and `PNEUMONIA` categories.
- Display of model scores and class-score comparisons.
- Grad-CAM visualizations to inspect image regions influencing a classification.
- Experimental 3D CT segmentation endpoint that returns a NIfTI mask.
- FastAPI backend with interactive API documentation.
- React + Vite frontend for image upload and result visualization.
- API tests covering health checks and invalid uploads.

## Technology stack

| Area | Technologies |
|---|---|
| Backend/API | Python, FastAPI |
| Deep learning | PyTorch, Torchvision, MONAI |
| Medical imaging/data | NiBabel, NumPy |
| Frontend | React, Vite, JavaScript, CSS |
| Testing | Pytest, Requests |

## Architecture

```text
React + Vite frontend
        |
        | HTTP requests
        v
FastAPI backend
   |             |
   v             v
X-ray model   CT segmentation
(ResNet18)    (experimental 3D U-Net)
   |
   v
Prediction scores + Grad-CAM explanation
```

## Project structure

```text
AI-Medical-Imaging/
├── backend/
│   ├── main.py
│   ├── gradcam.py
│   ├── train_model.py
│   ├── train_segmentation.py
│   ├── predict_segmentation.py
│   ├── evaluate_segmentation.py
│   └── test_api.py
├── data/
│   ├── chest_xray/
│   └── segmentation/
├── models/
│   ├── chest_xray_resnet18.pth
│   ├── class_names.json
│   └── lung_tumor_unet.pth
├── outputs/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── index.html
├── requirements.txt
└── README.md
```

Some data, model, or output files may not be present in every checkout. Check the repository contents before running training or inference workflows.

## Setup

### Prerequisites

- Python 3.11
- Node.js and npm
- Git (recommended)

### Backend

From the repository root, activate the existing virtual environment if you have one, then install dependencies:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the API:

```powershell
python -m uvicorn backend.main:app --reload
```

- API base URL: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs

### Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (typically http://localhost:5173).

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/upload` | Image upload |
| POST | `/predict` | Chest X-ray classification |
| POST | `/explain` | Grad-CAM visualization |
| POST | `/segment-ct` | Experimental CT segmentation |

Use the interactive API documentation at `/docs` to inspect request formats.

## Recorded experimental evaluation

### Chest X-ray classification

The recorded test-set results for the ResNet18 baseline were:

| Metric | Result |
|---|---:|
| Accuracy | 88.94% |
| Sensitivity | 99.23% |
| Specificity | 71.79% |
| Precision | 85.43% |

These are experimental results from the current evaluation, not evidence of clinical reliability. The specificity indicates false positives remain a concern.

### CT segmentation

A lightweight 3D U-Net was trained as a prototype using a limited subset of the available CT dataset. The recorded one-case Dice score was approximately **0.011**, indicating very poor overlap with the reference segmentation in that evaluation case. The segmentation output is experimental and must not be treated as reliable tumor localization.

## Testing

From the repository root:

```powershell
python -m pytest backend/test_api.py -v
```

The three API tests previously passed in the development environment. Rerun them after code changes.

Build the frontend:

```powershell
cd frontend
npm run build
```

The frontend production build previously succeeded in the development environment.

## Limitations and responsible use

- The classification model is an educational baseline, not a clinically validated diagnostic system.
- Grad-CAM highlights regions influencing a model prediction; it does not prove that disease is present in those regions.
- CT segmentation performance is currently poor and requires substantial improvement and multi-case validation.
- Results may be affected by dataset size, distribution, preprocessing, and model limitations.
- De-identify medical images. Do not upload identifiable patient information to an unapproved environment.
- Clinical use would require rigorous independent validation, specialist oversight, and applicable regulatory approval.

## Future improvements

- Improve segmentation preprocessing, training, and validation.
- Evaluate segmentation on multiple independent CT cases and report appropriate metrics.
- Expand automated tests for successful predictions and segmentation.
- Improve input validation, error handling, accessibility, and deployment documentation.

## Summary

This project demonstrates an end-to-end integration of deep-learning experimentation, explainability, medical-image processing, a Python API, and a React interface. Its current outputs are intended strictly for educational experimentation.
