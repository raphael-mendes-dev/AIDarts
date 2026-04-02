import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from backend.scorer import score_dart

router = APIRouter(prefix="/api")
log = logging.getLogger(__name__)

CAMERA_PROBE_MAX = 10
JPEG_QUALITY = 90
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dart_keypoints.onnx"
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"

# Change detection on warped frames (max per-camera diff vs held reference).
# Noise: ~1.1-1.3 per cam. Dart landing: persistent 3-7 (accumulates per dart).
DIFF_MOTION_THRESHOLD = 2.0
RECHECK_INTERVAL = 3.0  # seconds — periodic re-detection when darts are on the board

# Each camera capture thread runs at this rate; surplus frames are dropped.
_CAM_FPS = 15
_CAM_INTERVAL = 1.0 / _CAM_FPS

# ── Device settings (persisted to data/settings.json) ──

_settings_lock = threading.Lock()


def _read_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_settings(data: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))


@router.get("/settings")
def get_settings():
    with _settings_lock:
        return _read_settings()


@router.post("/settings")
def save_settings(body: dict[str, Any] = Body()):
    with _settings_lock:
        data = _read_settings()
        data.update(body)
        _write_settings(data)
    return {"ok": True}


# ── Detector (pre-loaded at startup) ──

def _load_detector():
    try:
        from backend.detector import DartDetector
        d = DartDetector(str(MODEL_PATH))
        log.info("Loaded model from %s (input %dx%d)", MODEL_PATH, d.img_size, d.img_size)
        return d
    except FileNotFoundError:
        log.warning("Model not found at %s — /api/detect will be unavailable", MODEL_PATH)
        return None


_detector = _load_detector()


# ── Helpers ──

def _capture_frame(index: int) -> np.ndarray:
    """Open camera, grab one frame, release. Fallback when grabber not running."""
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


def _validate_homography(H) -> bool:
    """Check that H is a valid 3x3 matrix with finite values."""
    try:
        a = np.array(H, dtype=np.float64)
        return a.shape == (3, 3) and np.all(np.isfinite(a))
    except (ValueError, TypeError):
        return False


def _norm_h_to_pixel(H, w, h):
    S_out = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]], dtype=np.float64)
    S_in = np.array([[1/w, 0, 0], [0, 1/h, 0], [0, 0, 1]], dtype=np.float64)
    return S_out @ np.array(H, dtype=np.float64) @ S_in


def _warp_crop_mask(frame: np.ndarray, H=None) -> np.ndarray:
    h, w = frame.shape[:2]
    if H is not None and _validate_homography(H):
        H_px = _norm_h_to_pixel(H, w, h)
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
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _run_detection(processed: list[np.ndarray], t_start: float, capture_ms: float = 0) -> dict:
    """Shared detection pipeline: BGR frames → inference → result dict with scores."""
    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in processed]
    count, kps, inference_ms = _detector.predict(pil_images)
    t_end = time.perf_counter()

    keypoints = []
    for i, (xn, yn, conf) in enumerate(kps):
        dart_score = score_dart(float(xn), float(yn))
        keypoints.append({
            "dart": i + 1,
            "x_norm": round(float(xn), 4),
            "y_norm": round(float(yn), 4),
            "confidence": round(float(conf), 4),
            **dart_score,
        })

    return {
        "count": count,
        "keypoints": keypoints,
        "time_ms": round(inference_ms, 1),
        "total_ms": round((t_end - t_start) * 1000, 1),
        "capture_ms": round(capture_ms, 1),
        "images": [_frame_to_b64jpg(f) for f in processed],
    }


# ── FrameSlot: latest-wins buffer with Condition ──

class FrameSlot:
    """Latest-wins frame buffer backed by threading.Condition.

    Stores both raw JPEG bytes (for MJPEG streaming passthrough) and a decoded
    numpy array (for OpenCV processing). Every waiting consumer is woken
    immediately when a new frame arrives — no missed-frame race.
    """

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._frame: bytes | None = None
        self._arr: np.ndarray | None = None
        self._seq: int = 0

    def put(self, data: bytes) -> None:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        with self._cond:
            self._frame = data
            self._arr = arr
            self._seq += 1
            self._cond.notify_all()

    def get_bytes(self, last_seq: int, timeout: float = 2.0) -> tuple[bytes | None, int]:
        """Block until seq advances past last_seq. Returns (frame_bytes, new_seq)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq == last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq
                self._cond.wait(remaining)
            return self._frame, self._seq

    def get_array(self) -> np.ndarray | None:
        with self._cond:
            return self._arr.copy() if self._arr is not None else None


# ── Per-camera capture thread ──

class _CamThread(threading.Thread):
    """Daemon thread that captures frames from one camera continuously.

    Tries linuxpy (V4L2 direct access, MJPEG passthrough, works on aarch64)
    first, then falls back to cv2 on platforms without V4L2 (Windows, macOS).
    Frame rate is capped at _CAM_FPS; surplus frames are dropped before any
    bytes copy, so backlog cannot accumulate.
    """

    def __init__(self, index: int, slot: FrameSlot, stop: threading.Event):
        super().__init__(daemon=True)
        self._index = index
        self._slot = slot
        self._stop = stop

    def run(self) -> None:
        try:
            self._run_linuxpy()
        except (ImportError, OSError) as exc:
            log.debug("Camera %d linuxpy unavailable (%s), using cv2", self._index, exc)
            self._run_cv2()

    def _run_linuxpy(self) -> None:
        from linuxpy.video.device import Device, VideoCapture, BufferType
        path = f"/dev/video{self._index}"
        with Device(path) as dev:
            try:
                dev.set_format(BufferType.VIDEO_CAPTURE, 640, 480, pixel_format="MJPG")
            except Exception as exc:
                log.debug("Camera %d set_format: %s", self._index, exc)
            with VideoCapture(dev) as cap:
                last = 0.0
                for frame in cap:
                    if self._stop.is_set():
                        break
                    now = time.monotonic()
                    if now - last < _CAM_INTERVAL:
                        continue  # drop frame before bytes() copy
                    last = now
                    self._slot.put(bytes(frame))

    def _run_cv2(self) -> None:
        cap = cv2.VideoCapture(self._index)
        try:
            last = 0.0
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                now = time.monotonic()
                if now - last < _CAM_INTERVAL:
                    continue
                last = now
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                self._slot.put(buf.tobytes())
        finally:
            cap.release()


# ── Background frame grabber with change detection ──

class FrameGrabber:
    def __init__(self):
        self._cameras: list[int] = []
        self._slots: dict[int, FrameSlot] = {}
        self._cam_stops: dict[int, threading.Event] = {}
        self._cam_threads: dict[int, _CamThread] = {}
        self._detect_thread: threading.Thread | None = None
        self._detect_stop = threading.Event()
        self._ref_frames: dict[int, np.ndarray] = {}
        self._change_pending = False
        self._auto_result: dict | None = None
        self._auto_result_id = 0
        self._auto_enabled = False
        self._homographies: dict[str, list] = {}
        self._last_detect_time: float = 0
        self._last_detect_count: int = 0

    def start(self, camera_indexes: list[int]):
        self.stop()
        self._cameras = camera_indexes
        self._change_pending = False
        self._auto_result = None
        self._auto_result_id = 0
        self._detect_stop.clear()

        for idx in camera_indexes:
            slot = FrameSlot()
            stop_ev = threading.Event()
            thread = _CamThread(idx, slot, stop_ev)
            self._slots[idx] = slot
            self._cam_stops[idx] = stop_ev
            self._cam_threads[idx] = thread
            thread.start()

        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._detect_thread.start()
        log.info("Frame grabber started for cameras %s", camera_indexes)

    def stop(self):
        self._detect_stop.set()
        if self._detect_thread and self._detect_thread.is_alive():
            self._detect_thread.join(timeout=3)
        self._detect_thread = None

        for stop_ev in self._cam_stops.values():
            stop_ev.set()
        for thread in self._cam_threads.values():
            thread.join(timeout=3)

        self._slots.clear()
        self._cam_stops.clear()
        self._cam_threads.clear()
        self._cameras = []
        self._ref_frames = {}
        self._auto_enabled = False
        log.info("Frame grabber stopped")

    @property
    def running(self):
        return bool(self._cam_threads) and any(t.is_alive() for t in self._cam_threads.values())

    def get_frame(self, index: int) -> np.ndarray | None:
        slot = self._slots.get(index)
        return slot.get_array() if slot else None

    def get_slot(self, index: int) -> FrameSlot | None:
        return self._slots.get(index)

    def get_cam_index(self, slot: int) -> int | None:
        """Return device index for camera slot (1-based)."""
        i = slot - 1
        return self._cameras[i] if 0 <= i < len(self._cameras) else None

    def set_homographies(self, homographies: dict[str, list]):
        self._homographies = homographies

    def set_auto(self, enabled: bool, homographies: dict[str, list] | None = None):
        self._auto_enabled = enabled
        if homographies is not None:
            self._homographies = homographies
        if enabled:
            self._ref_frames = {}
            self._change_pending = False
            self._auto_result = None
            log.info("Auto-detect enabled")
        else:
            log.info("Auto-detect disabled")

    def get_auto_result(self, after_id: int = 0) -> tuple[dict | None, int]:
        if self._auto_result_id > after_id:
            return self._auto_result, self._auto_result_id
        return None, self._auto_result_id

    def set_baseline(self):
        self._ref_frames = {}
        self._change_pending = False
        self._last_detect_count = 0

    def _detect_loop(self):
        while not self._detect_stop.wait(timeout=0.1):
            if self._auto_enabled:
                self._check_change()

    def _check_change(self):
        frames = {idx: slot.get_array() for idx, slot in self._slots.items()}
        frames = {k: v for k, v in frames.items() if v is not None}

        if not frames:
            return

        if self._change_pending:
            self._change_pending = False
            log.info("Running detection (all cameras refreshed)")
            self._run_auto_detect(frames)
            self._ref_frames = {}
            return

        warped = {}
        for i, idx in enumerate(self._cameras):
            f = frames.get(idx)
            if f is None:
                continue
            H = self._homographies.get(str(i + 1))
            warped[idx] = _warp_crop_mask(f, H)

        if not self._ref_frames:
            self._ref_frames = warped
            return

        diffs = []
        for idx in self._cameras:
            ref = self._ref_frames.get(idx)
            curr = warped.get(idx)
            if ref is None or curr is None or ref.shape != curr.shape:
                continue
            diffs.append(float(cv2.absdiff(ref, curr).mean()))

        if not diffs:
            return

        max_diff = max(diffs)
        log.debug("Frame diff max=%.2f (per cam: %s)", max_diff, [f"{d:.2f}" for d in diffs])

        if max_diff > DIFF_MOTION_THRESHOLD:
            log.info("Change detected (max_diff=%.1f) — will trigger next cycle", max_diff)
            self._change_pending = True
            return

        # Periodic re-detection when darts are on the board (for removal detection)
        if self._last_detect_count > 0 and (time.monotonic() - self._last_detect_time) > RECHECK_INTERVAL:
            log.info("Periodic recheck (last detected %d darts)", self._last_detect_count)
            self._change_pending = True

    def _run_auto_detect(self, frames: dict[int, np.ndarray]):
        if _detector is None:
            return
        try:
            t_start = time.perf_counter()
            processed = []
            for i, idx in enumerate(self._cameras):
                frame = frames.get(idx)
                if frame is None:
                    return
                H = self._homographies.get(str(i + 1))
                processed.append(_warp_crop_mask(frame, H))

            self._auto_result = _run_detection(processed, t_start)
            self._auto_result_id += 1
            self._last_detect_time = time.monotonic()
            self._last_detect_count = self._auto_result["count"]
            log.info("Auto-detect: %d dart(s), %.0fms",
                     self._auto_result["count"], self._auto_result["time_ms"])
        except Exception:
            log.exception("Auto-detect inference failed")


_grabber = FrameGrabber()

_MJPEG_BOUNDARY = b"frame"


# ── Endpoints ──

@router.get("/cameras")
def list_cameras():
    _grabber.stop()
    # Try linuxpy on Linux — auto-filters metadata-only /dev/videoN nodes
    try:
        from linuxpy.video.device import iter_video_capture_files, Device
        cameras = []
        for path in sorted(iter_video_capture_files()):
            try:
                with Device(path) as dev:
                    name = dev.info.card
                idx = int(str(path).replace("/dev/video", ""))
                cameras.append({"index": idx, "name": name, "width": 0, "height": 0})
            except (OSError, ValueError):
                pass
        return {"cameras": cameras}
    except ImportError:
        pass

    # Fallback: cv2 probe (Windows / non-V4L2 systems)
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
    if not (0 <= index < CAMERA_PROBE_MAX):
        raise HTTPException(400, "Invalid camera index")
    frame = _grabber.get_frame(index)
    if frame is None:
        frame = _capture_frame(index)
    if img_format == "png":
        _, buf = cv2.imencode(".png", frame)
        return Response(content=buf.tobytes(), media_type="image/png")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/stream/{slot}")
def stream_camera(slot: int):
    """Live MJPEG stream for camera slot 1/2/3 with homography applied when set."""
    if not (1 <= slot <= 3):
        raise HTTPException(400, "Slot must be 1, 2 or 3")
    cam_idx = _grabber.get_cam_index(slot)
    if cam_idx is None:
        raise HTTPException(404, "Camera slot not running")
    raw_slot = _grabber.get_slot(cam_idx)
    if raw_slot is None:
        raise HTTPException(404, "Stream not available")

    def generate():
        seq = 0
        while True:
            frame_bytes, seq = raw_slot.get_bytes(seq, timeout=2.0)
            if frame_bytes is None:
                continue
            H = _grabber._homographies.get(str(slot))
            if H:
                arr = raw_slot.get_array()
                if arr is not None:
                    warped = _warp_crop_mask(arr, H)
                    _, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    frame_bytes = buf.tobytes()
            yield (
                b"--" + _MJPEG_BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY.decode()}",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )


class CameraSet(BaseModel):
    cameras: dict[str, int]


@router.post("/grabber/start")
def grabber_start(req: CameraSet):
    indexes = [req.cameras[s] for s in ("1", "2", "3") if s in req.cameras]
    if not indexes:
        raise HTTPException(400, "No cameras specified")
    _grabber.start(indexes)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(_grabber.get_frame(i) is not None for i in indexes):
            break
        time.sleep(0.1)
    ready = all(_grabber.get_frame(i) is not None for i in indexes)
    if not ready:
        log.warning("Not all cameras ready after 5s: %s", indexes)
    return {"running": True, "cameras": indexes, "all_ready": ready}


@router.post("/grabber/stop")
@router.get("/grabber/stop")
def grabber_stop():
    _grabber.stop()
    return {"running": False}


class AutoDetectRequest(BaseModel):
    enabled: bool
    homographies: dict[str, list] = {}


@router.post("/grabber/auto")
def grabber_auto(req: AutoDetectRequest):
    _grabber.set_auto(req.enabled, req.homographies)
    return {"auto": req.enabled}


class HomographyUpdate(BaseModel):
    homographies: dict[str, list]


@router.post("/grabber/homographies")
def update_homographies(req: HomographyUpdate):
    """Update streaming homographies without changing auto-detect state."""
    _grabber.set_homographies(req.homographies)
    return {"ok": True}


@router.post("/grabber/baseline")
def grabber_baseline():
    _grabber.set_baseline()
    return {"ok": True}


@router.get("/grabber/result")
def grabber_result(after: int = 0):
    result, rid = _grabber.get_auto_result(after)
    if result:
        return {"result": result, "id": rid}
    return {"result": None, "id": rid}


class DetectRequest(BaseModel):
    cameras: dict[str, int]
    homographies: dict[str, list] = {}


@router.post("/detect")
def detect(req: DetectRequest):
    """Manual detection — uses grabbed frames if available."""
    try:
        t_start = time.perf_counter()
        if _detector is None:
            raise HTTPException(503, "Model not loaded")

        processed = []
        for slot in ("1", "2", "3"):
            cam_idx = req.cameras.get(slot)
            if cam_idx is None:
                raise HTTPException(400, f"Missing camera assignment for slot {slot}")
            frame = _grabber.get_frame(cam_idx)
            if frame is None:
                frame = _capture_frame(cam_idx)
            H = req.homographies.get(slot)
            processed.append(_warp_crop_mask(frame, H))

        capture_ms = (time.perf_counter() - t_start) * 1000
        _grabber.set_baseline()
        return _run_detection(processed, t_start, capture_ms)
    except HTTPException:
        raise
    except Exception:
        log.exception("Detection failed")
        raise HTTPException(500, "Detection failed — see server log")
