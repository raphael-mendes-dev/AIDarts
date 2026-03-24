import io
import logging
from pathlib import Path
from typing import Literal

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response
from PIL import Image

router = APIRouter(prefix="/api")
log = logging.getLogger(__name__)

CAMERA_PROBE_MAX = 10
JPEG_QUALITY = 90
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dart_keypoints.onnx"

_detector = None


def _get_detector():
    """Lazy-load detector so the server starts even without the model file."""
    global _detector
    if _detector is None:
        from backend.detector import DartDetector
        _detector = DartDetector(str(MODEL_PATH))
        log.info("Loaded model from %s (input %dx%d)", MODEL_PATH, _detector.img_size, _detector.img_size)
    return _detector


@router.get("/cameras")
def list_cameras():
    """Probe indexes 0-9 and return available cameras."""
    cameras = []
    for i in range(CAMERA_PROBE_MAX):
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
def snapshot(index: int, img_format: Literal["jpg", "png"] = "jpg"):
    """Grab a single frame from the given camera index."""
    if not (0 <= index < CAMERA_PROBE_MAX):
        raise HTTPException(400, "Invalid camera index")
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise HTTPException(404, "Camera not available")
        ret, frame = cap.read()
        if not ret:
            raise HTTPException(500, "Failed to capture frame")
    finally:
        cap.release()
    if img_format == "png":
        _, buf = cv2.imencode(".png", frame)
        return Response(content=buf.tobytes(), media_type="image/png")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/detect")
def detect(
    cam1: UploadFile = File(...),
    cam2: UploadFile = File(...),
    cam3: UploadFile = File(...),
):
    """Run dart keypoint detection on 3 warped camera images."""
    detector = _get_detector()
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
