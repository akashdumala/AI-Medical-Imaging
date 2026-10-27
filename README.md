# AI-Assisted Medical Imaging and Diagnosis System

## 1. Project Overview

This project is an educational medical-imaging prototype that combines deep learning, a REST API, and a React web interface to explore medical image classification, visual explanations, and experimental CT segmentation.

**Important:** This project is for research and educational demonstration only. It is not a medical device and must not be used to diagnose patients or guide treatment.

## 2. Features

- Chest X-ray classification into NORMAL and PNEUMONIA categories.
- Model score display and class-score comparison.
- Grad-CAM visualization to illustrate image regions influencing a classification.
- Experimental 3D CT segmentation endpoint that returns a NIfTI mask.
- FastAPI backend with interactive API documentation.
- React and Vite frontend for uploading and viewing supported images.
- Automated API tests for health checks and invalid uploads.

## 3. Technology Stack

### Backend
- Python
- FastAPI
- PyTorch and Torchvision
- MONAI
- NiBabel
- NumPy

### Frontend
- React
- Vite
- JavaScript
- CSS
- Lucide React

### Testing
- Pytest
- Requests

## 4. Project Structure

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

Some files and folders may be absent if they have not been created or retained locally.

## 5. Installation

### Prerequisites

- Python 3.11
- Node.js and npm
- Git (recommended)

### Backend setup

Open a terminal in the project root and activate the existing virtual environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the backend:

```powershell
python -m uvicorn backend.main:app --reload
```

API base URL:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### Frontend setup

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, typically:

`http://localhost:5173`

## 6. API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Root endpoint |
| `GET /health` | Health check |
| `POST /upload` | Image upload endpoint |
| `POST /predict` | Chest X-ray classification |
| `POST /explain` | Grad-CAM visualization |
| `POST /segment-ct` | Experimental CT segmentation |

Use the interactive API documentation to inspect request formats and test the available endpoints.

## 7. Model Evaluation

### Chest X-ray classification

The recorded test-set results for the ResNet18 baseline were:

- Accuracy: 88.94%
- Sensitivity: 99.23%
- Specificity: 71.79%
- Precision: 85.43%

These results are from the current experimental evaluation. They do not establish clinical reliability, and the relatively low specificity indicates that false-positive classifications remain a concern.

### CT segmentation

A lightweight 3D U-Net was trained as a prototype using a limited subset of the available CT dataset.

The recorded one-case Dice score was approximately 0.011. This indicates very poor overlap with the reference segmentation on that evaluation case. The model must be treated as experimental and its output must not be interpreted as a reliable tumor localization.

## 8. Testing

From the project root, run:

```powershell
python -m pytest backend/test_api.py -v
```

The three API tests previously passed in the development environment. Rerun them after making code changes.

Build the frontend for production:

```powershell
cd frontend
npm run build
```

The frontend production build has also previously succeeded in the development environment.

## 9. Limitations and Safety

- The classification model is an educational baseline, not a clinically validated diagnostic system.
- Grad-CAM highlights model-relevant image regions but does not prove that a disease or lesion exists in those regions.
- CT segmentation performance is currently poor and requires substantial further development and evaluation.
- Results may be affected by dataset size, data distribution, preprocessing, and model limitations.
- Uploaded medical images should be de-identified. Do not upload identifiable patient information to an unapproved environment.
- Clinical use requires appropriate datasets, rigorous independent validation, specialist oversight, and applicable regulatory approval.

## 10. Future Improvements

- Improve segmentation preprocessing, training, and validation.
- Evaluate on multiple independent CT cases and report Dice and other appropriate metrics.
- Add input validation and clearer error handling.
- Add automated tests for successful predictions and segmentation.
- Containerize the application and document deployment.
- Improve frontend visualization and accessibility.

## 11. Conclusion

This prototype demonstrates how deep-learning models, explainability visualizations, a Python API, and a React frontend can be integrated into a medical-imaging research application. Its current outputs are intended for educational experimentation only.
