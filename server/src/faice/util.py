"""Utility functions for face detection and emotion classification."""

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from faice.config import (
    CLASSIFIER_WEIGHT_PATH,
    DETECTOR_PATH,
    DEVICE,
    EMOTION,
    EMOTION_MAP,
    IMAGE_SIZE,
)
from faice.model_arch import EmotionClassifier


def load_detector() -> YOLO:
    return YOLO(str(DETECTOR_PATH), task="detect")


def load_classifier() -> EmotionClassifier:
    model = EmotionClassifier()
    model.load_state_dict(torch.load(str(CLASSIFIER_WEIGHT_PATH), map_location=DEVICE))
    model.eval()
    return model


def crop_face(model, image_orig: Image.Image):
    """Detect the first face. Returns (cropped, bbox, original)."""
    result = model(image_orig)[0]
    boxes = result.boxes

    if len(boxes) == 0:
        return None, None, image_orig

    box = boxes[0]
    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
    image_cropped = image_orig.crop((x_min, y_min, x_max, y_max))

    print(f"📦 BBox: {x_min}, {y_min}, {x_max}, {y_max}")
    return image_cropped, (x_min, y_min, x_max, y_max), image_orig


transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def predict_expression(model: EmotionClassifier, image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(image_tensor)
        confidences = torch.softmax(outputs, dim=1)[0]

    idx = int(torch.argmax(confidences).item())
    emotion = EMOTION[idx]

    all_confidences = {
        EMOTION_MAP[EMOTION[i]]: round(confidences[i].item(), 2) for i in range(len(EMOTION))
    }

    return EMOTION_MAP[emotion], round(confidences[idx].item(), 2), all_confidences


def draw_emotion_box(frame, bbox, emotion_label: str, confidence: float):
    if bbox is None:
        return frame

    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        f"{emotion_label} {confidence:.2f}",
        (x1, y1 - 10),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return frame
