"""Runtime constants for the emotion pipeline."""

from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
DETECTOR_PATH = MODELS_DIR / "yolo.onnx"
CLASSIFIER_WEIGHT_PATH = MODELS_DIR / "efficientnet_b3.pth"

EMOTION = ["anger", "happy", "panic", "sadness"]
EMOTION_MAP = {
    "anger": "angry",
    "happy": "happy",
    "panic": "awe",
    "sadness": "sad",
    "blank": "blank",
}
IMAGE_SIZE = 300
