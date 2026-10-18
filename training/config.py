"""
Configuration settings for State Farm dataset and training.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "state_farm"
MODELS_DIR = BASE_DIR / "models"

# 10 State Farm Distraction Classes
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

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
