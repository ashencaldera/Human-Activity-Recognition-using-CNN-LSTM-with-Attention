"""
data/download_data.py
─────────────────────
Downloads and extracts the UCI HAR dataset automatically.
Run: python data/download_data.py
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)
DATA_DIR = Path(__file__).parent
ZIP_PATH = DATA_DIR / "UCI_HAR.zip"
EXTRACT_DIR = DATA_DIR / "UCI HAR Dataset"


def download_uci_har():
    if EXTRACT_DIR.exists():
        print(f"[INFO] Dataset already present at: {EXTRACT_DIR}")
        return str(EXTRACT_DIR)

    print("[INFO] Downloading UCI HAR Dataset (~60 MB)…")
    urllib.request.urlretrieve(UCI_HAR_URL, ZIP_PATH)
    print("[INFO] Extracting…")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    ZIP_PATH.unlink(missing_ok=True)
    print(f"[INFO] Done → {EXTRACT_DIR}")
    return str(EXTRACT_DIR)


if __name__ == "__main__":
    download_uci_har()
