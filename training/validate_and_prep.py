"""
Dataset validation and preparation utility for Kaggle State Farm Distracted Driver Detection.
Checks folder layout for classes c0-c9 and sample availability.
"""
from pathlib import Path
import sys

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "state_farm"

# 10 State Farm Classes
CLASSES = {
    "c0": "safe driving",
    "c1": "texting - right",
    "c2": "talking on the phone - right",
    "c3": "texting - left",
    "c4": "talking on the phone - left",
    "c5": "operating the radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger",
}


def validate_dataset():
    """Validates presence of raw train images for classes c0-c9."""
    train_dir = DATASET_DIR / "train"
    if not train_dir.exists():
        print(f"[!] Dataset folder not found: {train_dir}")
        print("    Place State Farm dataset train folder under: datasets/state_farm/train/")
        return False

    missing = []
    found = {}
    for c_id, c_name in CLASSES.items():
        c_folder = train_dir / c_id
        if not c_folder.exists():
            missing.append(c_id)
        else:
            img_count = len(list(c_folder.glob("*.jpg")))
            found[c_id] = (c_name, img_count)

    if missing:
        print(f"[!] Missing classes in {train_dir}: {', '.join(missing)}")
        return False

    print(f"[OK] State Farm dataset found at {train_dir}:")
    for c_id, (name, count) in found.items():
        print(f"    - {c_id} ({name}): {count} images")
    return True


if __name__ == "__main__":
    validate_dataset()
