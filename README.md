# AIDarts

AI-powered dart scoring system using 3 synchronized cameras to detect and score darts on a dartboard.

Developed on a **Qualcomm QRB2210 (Arduino UNO Q, 2GB)** with **3x OV9732 USB cameras** on a USB hub and **Autodarts 3D-printed camera mounts**. Works on any machine with Python 3.10+ and USB cameras.

## Requirements

### Hardware

- **3 USB cameras** pointed at the dartboard from different angles
- A **dartboard** (PDC standard, 451mm outer diameter)
- A host machine running Python 3.10+ (Linux, macOS, or Windows)

### Software

- Python 3.10+
- A modern web browser (Chrome, Edge, or Firefox) on any device on the same network

### Model

The ONNX model file (`dart_keypoints.onnx`) is **not included** in the repository. Place it in `models/dart_keypoints.onnx` before using the detection features.

## Installation

```bash
git clone https://github.com/Palhera/AIDarts.git
cd AIDarts
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp /path/to/dart_keypoints.onnx models/
```

## Running

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://<host-ip>:8000` from any device on the local network.

## How It Works

### Overview

No build step, no bundler, no framework. The frontend is vanilla HTML/CSS/JS (ES modules) served as static files. The backend is a FastAPI app with sync endpoints running in a threadpool.

| Page | URL | Purpose |
|------|-----|---------|
| Home | `/` | Game mode selection, player count |
| Settings | `/settings` | Camera scan, assignment, calibration with wire overlay, fusion preview |
| Test | `/test` | Keypoint detection testing (manual + auto-detect) |

### Camera Capture

Each camera runs in its own dedicated daemon thread with a persistent connection, so all 3 capture simultaneously. On Linux (including the QRB2210), threads use **linuxpy** for direct V4L2 access — the kernel's USB scheduler handles bus arbitration between devices on the same hub, and MJPEG frames are forwarded from the kernel buffer with zero re-encode cost. On Windows/macOS, threads fall back to `cv2.VideoCapture`.

Each thread writes its latest frame into a `FrameSlot` (a latest-wins buffer backed by `threading.Condition`), which wakes all consumers — stream clients and the change-detection thread — simultaneously on every new frame.

### Calibration

For each camera, the user aligns a wire SVG overlay to the dartboard by dragging 4 handles. This computes a homography (4-point DLT) that maps the camera view to a standardized dartboard reference. Rotation buttons allow 18-degree coarse alignment. The live camera tiles in the Settings page show the homography-corrected view in real time via MJPEG streams. All state is persisted server-side in `data/settings.json`.

### Detection Pipeline

The browser sends camera indexes and homography matrices to `POST /api/detect`. The server reads the latest frame from each camera's slot, warps each with `cv2.warpPerspective`, crops to a centered square, applies a circular board mask, then runs ONNX inference. Returns keypoint positions, confidence scores, timing, and JPEG previews.

### Auto-Detect

A dedicated change-detection thread continuously reads the latest frames from all camera slots and compares warped frames against a held reference via `cv2.absdiff`. When change is detected (dart landed), it waits one more cycle for all cameras to refresh, then triggers inference automatically. The frontend polls for results every 250ms.

### MJPEG Streaming

`GET /api/stream/{slot}` serves a live `multipart/x-mixed-replace` MJPEG stream for each camera slot with the calibration homography applied server-side. The Settings page uses these streams directly in `<img>` elements — no JavaScript required for display.

## Limitations

- **CPU-only inference** — the QRB2210 has no CUDA GPU, so ONNX runs on CPU. A CUDA-capable host would be significantly faster
- **3 cameras required** — the model expects exactly 3 views
