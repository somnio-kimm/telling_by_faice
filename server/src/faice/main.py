"""FastAPI entry point. Serves emotion prediction and scenario generation endpoints to the Godot game."""

import base64
import traceback
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

from faice.prompt_chain import evaluate_response, generate_scenario
from faice.util import (
    crop_face,
    draw_emotion_box,
    load_classifier,
    load_detector,
    predict_expression,
)

camera = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera
    camera = cv2.VideoCapture(0)
    print("📸 Camera started")
    yield
    camera.release()
    print("🔒 Camera released")


app = FastAPI(lifespan=lifespan)
detection_model = load_detector()
classification_model = load_classifier()


@app.get("/")
def root():
    return {"message": "Facial Expression API is running."}


@app.post("/predict/")
async def predict():
    ret, frame = camera.read()
    if not ret:
        return JSONResponse({"error": "Webcam capture failed"}, status_code=500)

    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        image_cropped, bbox, image_orig = crop_face(detection_model, pil_image)
        frame_to_draw = cv2.cvtColor(np.array(image_orig), cv2.COLOR_RGB2BGR)

        if image_cropped and bbox:
            emotion, confidence, all_confidences = predict_expression(classification_model, image_cropped)
            frame_to_draw = draw_emotion_box(frame_to_draw, bbox, emotion, confidence)
        else:
            emotion, confidence = "no_face", 0.0
            all_confidences = None
            print("⚠️ No face detected")

        frame_to_draw = cv2.resize(frame_to_draw, (300, 300))
        success, buffer = cv2.imencode(".jpg", frame_to_draw)
        if not success:
            return JSONResponse({"error": "Failed to encode image"}, status_code=500)

        img_bytes = base64.b64encode(buffer).decode("utf-8")
        return JSONResponse(
            {
                "emotion": emotion,
                "confidence": confidence,
                "bbox": bbox,
                "image": img_bytes,
                "all_confidences": all_confidences,
            }
        )

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/generate_scenario")
async def get_scenario():
    try:
        result = generate_scenario()
        return JSONResponse({"scenario": result["scenario"]})
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/evaluate_response")
async def evaluate_endpoint(payload: dict):
    try:
        scenario = payload.get("scenario", "")
        user_response = payload.get("user_response", "")
        return evaluate_response(scenario, user_response)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)
