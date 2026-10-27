
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200


def test_invalid_xray_upload():
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("invalid.txt", b"not an image", "text/plain")},
        timeout=30,
    )
    assert response.status_code in (400, 415, 422)


def test_invalid_ct_upload():
    response = requests.post(
        f"{BASE_URL}/segment-ct",
        files={"file": ("invalid.nii.gz", b"not a CT scan", "application/gzip")},
        timeout=30,
    )
    assert response.status_code in (400, 415, 422)
