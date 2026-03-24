import base64
import logging
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

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


def _capture_frame(index: int) -> np.ndarray:
    """Open camera, grab one frame, release. Sequential-safe for USB hubs."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise HTTPException(404, f"Camera {index} not available")
        ret, frame = cap.read()
        if not ret:
            raise HTTPException(500, f"Camera {index} failed to capture")
    finally:
        cap.release()
    return frame


def _norm_h_to_pixel(H, w, h):
    """Convert normalized (0-1) homography to pixel-space for cv2.warpPerspective."""
    S_out = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]], dtype=np.float64)
    S_in = np.array([[1/w, 0, 0], [0, 1/h, 0], [0, 0, 1]], dtype=np.float64)
    H_np = np.array(H, dtype=np.float64)
    return S_out @ H_np @ S_in


def _warp_crop_mask(frame: np.ndarray, H=None) -> np.ndarray:
    """Apply homography warp, crop to centered square, apply circular mask."""
    h, w = frame.shape[:2]

    if H is not None:
        H_px = _norm_h_to_pixel(H, w, h)
        # Stored H maps output_norm→input_norm (inverse warp)
        frame = cv2.warpPerspective(frame, H_px, (w, h),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    sz = min(w, h)
    x0 = (w - sz) // 2
    y0 = (h - sz) // 2
    square = frame[y0:y0+sz, x0:x0+sz]

    mask = np.zeros((sz, sz), dtype=np.uint8)
    cv2.circle(mask, (sz // 2, sz // 2), sz // 2, 255, thickness=-1, lineType=cv2.LINE_AA)
    out = np.zeros_like(square)
    out[mask > 0] = square[mask > 0]
    return out


def _frame_to_b64jpg(frame: np.ndarray) -> str:
    """Encode BGR frame as base64 JPEG for JSON response."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode("ascii")


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
    frame = _capture_frame(index)
    if img_format == "png":
        _, buf = cv2.imencode(".png", frame)
        return Response(content=buf.tobytes(), media_type="image/png")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


class DetectRequest(BaseModel):
    cameras: dict[str, int]       # {"1": cam_idx, "2": cam_idx, "3": cam_idx}
    homographies: dict[str, list] = {}  # {"1": [[...],[...],[...]], ...}


@router.post("/detect")
def detect(req: DetectRequest):
    """Capture 3 cameras, warp, crop, mask, run detection — all server-side."""
    try:
        t_start = time.perf_counter()
        detector = _get_detector()

        # Open/read/close each camera sequentially (USB hub constraint)
        processed = []
        for slot in ("1", "2", "3"):
            cam_idx = req.cameras.get(slot)
            if cam_idx is None:
                raise HTTPException(400, f"Missing camera assignment for slot {slot}")
            frame = _capture_frame(cam_idx)
            H = req.homographies.get(slot)
            processed.append(_warp_crop_mask(frame, H))

        t_capture = time.perf_counter()

        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in processed]
        count, kps, inference_ms = detector.predict(pil_images)

        t_end = time.perf_counter()

        images_b64 = [_frame_to_b64jpg(f) for f in processed]

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
            "time_ms": round(inference_ms, 1),
            "total_ms": round((t_end - t_start) * 1000, 1),
            "capture_ms": round((t_capture - t_start) * 1000, 1),
            "images": images_b64,
        }
    except HTTPException:
        raise
    except Exception:
        log.exception("Detection failed")
        raise HTTPException(500, "Detection failed — see server log")
