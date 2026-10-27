
from pathlib import Path
from monai.apps import DecathlonDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "segmentation"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Dataset location:", DATA_DIR)
print("Downloading Medical Segmentation Decathlon Task06_Lung...")
print("This is a large download; do not interrupt it.")

dataset = DecathlonDataset(
    root_dir=str(DATA_DIR),
    task="Task06_Lung",
    section="training",
    download=True,
    transform=(),
    cache_rate=0.0,
    num_workers=0,
)

print("Download and dataset initialization completed.")
print("Number of training cases:", len(dataset))
print("Dataset location:", DATA_DIR / "Task06_Lung")
