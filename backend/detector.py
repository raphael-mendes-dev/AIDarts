"""
Modular dart keypoint detector — copy this file + ONNX model into any project.

Detects dart tip positions on a dartboard from a 3-camera rig. The model is a
U-Net v2 (depthwise separable) that takes three RGB images (one per camera),
concatenates them into a 9-channel tensor, and outputs:
  - a dart count classification (0, 1, 2, or 3 darts)
  - a heatmap from which dart tip keypoints are extracted via peak detection
    with subpixel refinement

Dependencies:
    pip install numpy onnxruntime Pillow scipy

Usage:
    from detector import DartDetector

    detector = DartDetector("dart_keypoints.onnx")

    # cam1, cam2, cam3 are PIL.Image.Image objects (any size, any mode)
    count, keypoints, elapsed_ms = detector.predict([cam1, cam2, cam3])

    # count:      int         — number of darts detected (0–3)
    # keypoints:  list[tuple] — [(x_norm, y_norm, confidence), ...] per dart
    #             x_norm/y_norm are in [0, 1]; multiply by image width/height
    #             to get pixel coordinates
    # elapsed_ms: float       — ONNX inference time (excludes preprocessing)
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

log = logging.getLogger(__name__)

Keypoint = Tuple[float, float, float]  # (x_norm, y_norm, confidence)

# ImageNet normalization repeated for 9 channels (3 cameras x 3 RGB channels)
NORM_MEAN = np.array([0.485, 0.456, 0.406] * 3, dtype=np.float32).reshape(9, 1, 1)
NORM_STD = np.array([0.229, 0.224, 0.225] * 3, dtype=np.float32).reshape(9, 1, 1)


class DartDetector:
    """ONNX-based dart tip keypoint detector for a 3-camera dartboard rig.

    Attributes:
        session:  The ONNX Runtime inference session.
        img_size: Spatial resolution the model expects (read from the model,
                  defaults to 192 if dynamic).
    """

    def __init__(
        self,
        model_path: str,
        intra_threads: int = 4,
        inter_threads: int = 1,
    ):
        """Load the ONNX model and create an inference session.

        Args:
            model_path:     Path to the .onnx model file.
            intra_threads:  Threads for intra-op parallelism (default 4).
            inter_threads:  Threads for inter-op parallelism (default 1).

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = inter_threads
        opts.intra_op_num_threads = intra_threads

        self.session = ort.InferenceSession(str(model_path.resolve()), sess_options=opts)
        input_shape = self.session.get_inputs()[0].shape
        if isinstance(input_shape[2], int):
            self.img_size = input_shape[2]
        else:
            self.img_size = 192
            log.warning("Model has dynamic input shape %s, defaulting to %d", input_shape, self.img_size)

    def _build_input(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Convert 3 PIL images into a normalized (1, 9, H, W) float32 tensor.

        Each image is converted to RGB, resized to img_size x img_size,
        scaled to [0, 1], and ImageNet-normalized. The three 3-channel arrays
        are concatenated into a single 9-channel tensor.
        """
        channels = []
        for img in pil_images:
            img = img.convert("RGB").resize(
                (self.img_size, self.img_size), Image.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 255.0
            channels.append(arr.transpose(2, 0, 1))
        x = np.concatenate(channels, axis=0)
        x = (x - NORM_MEAN) / NORM_STD
        return x[np.newaxis, ...]

    @staticmethod
    def _subpixel_refine(hm_smooth, py, px, H, W):
        """Apply Taylor-expansion subpixel refinement around a peak.

        Computes the 2D Hessian at (py, px) and shifts the peak by up to
        +/-0.5 pixels for sub-pixel accuracy. Returns (dx, dy) offsets.
        """
        if not (1 <= px < W - 1 and 1 <= py < H - 1):
            return 0.0, 0.0
        gx = (hm_smooth[py, px + 1] - hm_smooth[py, px - 1]) / 2.0
        gy = (hm_smooth[py + 1, px] - hm_smooth[py - 1, px]) / 2.0
        dxx = hm_smooth[py, px + 1] - 2 * hm_smooth[py, px] + hm_smooth[py, px - 1]
        dyy = hm_smooth[py + 1, px] - 2 * hm_smooth[py, px] + hm_smooth[py - 1, px]
        dxy = (
            hm_smooth[py + 1, px + 1]
            - hm_smooth[py + 1, px - 1]
            - hm_smooth[py - 1, px + 1]
            + hm_smooth[py - 1, px - 1]
        ) / 4.0
        det = dxx * dyy - dxy * dxy
        if abs(det) > 1e-6:
            return (
                np.clip(-(dyy * gx - dxy * gy) / det, -0.5, 0.5),
                np.clip(-(dxx * gy - dxy * gx) / det, -0.5, 0.5),
            )
        return 0.0, 0.0

    def predict(
        self, pil_images: List[Image.Image]
    ) -> Tuple[int, List[Keypoint], float]:
        """Run dart detection on 3 camera images.

        Args:
            pil_images: List of exactly 3 PIL images (cam1, cam2, cam3).
                        Any size/mode accepted (converted internally).

        Pipeline:
            1. Preprocess: resize, normalize, concatenate into 9-channel tensor
            2. ONNX inference: produces count logits + heatmap
            3. Post-process: Gaussian smooth -> non-max suppression -> top-k
               peaks -> subpixel refinement -> normalized coordinates

        Returns:
            Tuple of (count, keypoints, elapsed_ms):
                count:      Number of darts detected (0–3).
                keypoints:  List of (x_norm, y_norm, confidence) tuples,
                            length == count. Coordinates are in [0, 1];
                            multiply by image width/height for pixels.
                elapsed_ms: ONNX inference wall time in milliseconds
                            (excludes pre/post-processing).
        """
        if len(pil_images) != 3:
            raise ValueError(f"Expected exactly 3 images, got {len(pil_images)}")
        x = self._build_input(pil_images)

        t0 = time.perf_counter()
        count_logits, heatmap = self.session.run(None, {"input": x})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Count head: pick class with highest logit (0, 1, 2, or 3)
        count = int(count_logits[0].argmax())

        # Heatmap head: single-channel spatial map of dart-tip likelihood
        hm = heatmap[0, 0]
        H, W = hm.shape

        # Peak detection: smooth, find local maxima, take top-k
        hm_s = gaussian_filter(hm, sigma=0.5)
        local_max = maximum_filter(hm_s, size=9)
        peaks = (hm_s == local_max) * hm_s
        top_idx = np.argsort(peaks.reshape(-1))[::-1][:count]

        kps: List[Keypoint] = []
        for idx in top_idx:
            py, px = divmod(int(idx), W)
            conf = float(hm[py, px])
            dx, dy = self._subpixel_refine(hm_s, py, px, H, W)
            kps.append(((px + dx) / (W - 1), (py + dy) / (H - 1), conf))

        return count, kps, elapsed_ms
