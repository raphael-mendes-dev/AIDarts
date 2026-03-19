import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/api")


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
def snapshot(index: int):
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
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")
