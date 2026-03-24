import io
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response
from PIL import Image

from backend.detector import DartDetector

router = APIRouter(prefix="/api")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dart_keypoints.onnx"
detector = DartDetector(str(MODEL_PATH))


@router.get("/cameras")
def list_cameras():
    """Probe indexes 0-9 and return available cameras."""
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({"index": i, "name": f"Camera {i}", "width": w, "height": h})
        finally:
            cap.release()
    return {"cameras": cameras}


@router.get("/cameras/{index}/snapshot")
def snapshot(index: int, format: str = "jpg"):
    """Grab a single frame from the given camera index."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise HTTPException(404, "Camera not available")
        ret, frame = cap.read()
        if not ret:
            raise HTTPException(500, "Failed to capture frame")
    finally:
        cap.release()
    if format == "png":
        _, buf = cv2.imencode(".png", frame)
        return Response(content=buf.tobytes(), media_type="image/png")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/detect")
def detect(
    cam1: UploadFile = File(...),
    cam2: UploadFile = File(...),
    cam3: UploadFile = File(...),
):
    """Run dart keypoint detection on 3 warped camera images."""
    pil_images = []
    for f in (cam1, cam2, cam3):
        pil_images.append(Image.open(io.BytesIO(f.file.read())).convert("RGB"))

    count, kps, elapsed_ms = detector.predict(pil_images)

    keypoints = [
        {
            "dart": i + 1,
            "x_norm": round(float(xn), 4),
            "y_norm": round(float(yn), 4),
            "confidence": round(float(conf), 4),
        }
        for i, (xn, yn, conf) in enumerate(kps)
    ]

    return {
        "count": count,
        "keypoints": keypoints,
        "time_ms": round(elapsed_ms, 1),
    }
