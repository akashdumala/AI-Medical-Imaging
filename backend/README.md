# AI-Assisted Medical Imaging and Diagnosis System

An educational medical-imaging research prototype developed using Python, FastAPI, PyTorch, MONAI, and React. The system demonstrates chest X-ray classification, Grad-CAM visualization, and experimental 3D CT segmentation through a web dashboard.

> **Disclaimer:** This is a research and educational prototype, not a clinically validated medical device. Outputs must not be used for diagnosis or treatment decisions.

## Features

- **Chest X-ray classification:** Classifies uploaded images as NORMAL or PNEUMONIA.
- **Grad-CAM visualization:** Generates a heatmap overlay to visualize model attention.
- **Experimental CT segmentation:** Processes NIfTI CT volumes and returns a downloadable segmentation mask.
- **REST API:** FastAPI endpoints for model inference.
- **Web dashboard:** React interface for image uploads, predictions, visualizations, and downloads.
- **Automated tests:** Pytest tests for API health and invalid file uploads.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI, Uvicorn |
| X-ray model | PyTorch, ResNet18 |
| Explainability | Grad-CAM |
| CT segmentation | MONAI, 3D U-Net |
| Medical image formats | NiBabel, NIfTI |
| Testing | Pytest, Requests |

## Project Structure

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
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   └── package.json
├── data/
│   ├── chest_xray/
│   └── segmentation/
├── models/
├── outputs/
└── README.md
```

## Prerequisites

- Python 3.11
- Node.js and npm
- A Windows, Linux, or macOS development environment
- Sufficient memory and storage for medical imaging datasets

## Setup and Run

### 1. Backend

From the project root, activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install required Python packages if needed:

```powershell
python -m pip install fastapi uvicorn python-multipart torch torchvision monai nibabel numpy pillow requests pytest
```

Start the API:

```powershell
python -m uvicorn backend.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### 2. Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL shown by Vite, normally:

http://localhost:5173

If the frontend cannot call the backend, check that FastAPI CORS middleware allows the frontend origin.

### 3. Run Automated Tests

From the project root, with the backend running:

```powershell
python -m pytest backend/test_api.py -v
```

The current API test suite covers the health endpoint and rejection of invalid X-ray and CT uploads.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Chest X-ray classification |
| `/explain` | POST | Grad-CAM visualization |
| `/segment-ct` | POST | Experimental CT segmentation |

Use the Swagger documentation to inspect the actual response schemas and test each endpoint.

## Model Evaluation and Limitations

### Chest X-ray classifier

The current baseline test results are:

- Accuracy: 88.94%
- Sensitivity: 99.23%
- Specificity: 71.79%
- Precision: 85.43%

These results are specific to the current test split and do not establish clinical performance. The relatively low specificity means false-positive predictions remain a concern. Model scores are not calibrated probabilities.

### CT segmentation

The current 3D U-Net implementation has successfully processed a CT volume and generated a downloadable NIfTI mask. However, the initial smoke-test training used only four cases for two epochs.

An initial evaluation on one case produced a Dice score of approximately 0.0105. This indicates very poor segmentation overlap. The CT model must therefore be considered experimental and its output must not be presented as reliable tumor detection.

Further work is required, including proper training, case-level validation, threshold and preprocessing review, and evaluation across independent test cases.

### Grad-CAM

Grad-CAM provides a visualization of model attention. It does not prove that the model has identified a disease or localized a lesion.

## Future Improvements

- Train and evaluate the CT segmentation model using a suitable training protocol.
- Add automated tests for successful predictions and Grad-CAM output.
- Improve input validation, error messages, and logging.
- Add visualization of CT slices and segmentation overlays.
- Add reproducible training configurations and model versioning.
- Containerize the application with Docker.
- Document dataset sources, licensing, and preprocessing.
- Add continuous integration and a reproducible evaluation report.

## Responsible Use

Use only appropriately licensed research datasets and avoid uploading identifiable patient information. This project is intended for education and experimentation, not clinical deployment.

## Project Status

The frontend and core inference endpoints are operational in the local development environment. Automated API tests currently pass. Model validation, particularly for CT segmentation, remains incomplete.
