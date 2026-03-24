/* Shared utilities for AIDarts frontend */

export const EPS = 1e-12;

export const LS = {
  DC:   "aidarts_dc_enabled",
  CAMS: "aidarts_cam_assign",
  CAL:  "aidarts_calib",
  DET:  "aidarts_detected",
  H:    "aidarts_homography",
};

export const $ = (s, p) => (p || document).querySelector(s);
export const $$ = (s, p) => [...(p || document).querySelectorAll(s)];

export function lsGet(k, fb) {
  try { return JSON.parse(localStorage.getItem(k)) || fb; } catch { return fb; }
}

export function lsSet(k, v) {
  try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { console.warn("localStorage write failed:", k, e); }
}

export function loadImg(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Image load failed: " + url));
    img.src = url;
  });
}

/**
 * Bilinear homography warp. Returns a canvas (caller can .toDataURL() if needed).
 * Pre-fills with opaque black so unmapped pixels match training data.
 */
export function warpWithH(img, H) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const srcCvs = document.createElement("canvas");
  srcCvs.width = w; srcCvs.height = h;
  srcCvs.getContext("2d").drawImage(img, 0, 0);
  const sd = srcCvs.getContext("2d").getImageData(0, 0, w, h).data;

  const outCvs = document.createElement("canvas");
  outCvs.width = w; outCvs.height = h;
  const outCtx = outCvs.getContext("2d");
  outCtx.fillStyle = "#000";
  outCtx.fillRect(0, 0, w, h);
  const out = outCtx.getImageData(0, 0, w, h);
  const od = out.data;
  const wm1 = w - 1, hm1 = h - 1;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const nx = x / w, ny = y / h;
      const denom = H[2][0] * nx + H[2][1] * ny + H[2][2];
      if (Math.abs(denom) < EPS) continue;
      const fx = ((H[0][0] * nx + H[0][1] * ny + H[0][2]) / denom) * w;
      const fy = ((H[1][0] * nx + H[1][1] * ny + H[1][2]) / denom) * h;
      if (fx < 0 || fx > wm1 || fy < 0 || fy > hm1) continue;

      const x0 = fx | 0, y0 = fy | 0;
      const x1 = Math.min(x0 + 1, wm1), y1 = Math.min(y0 + 1, hm1);
      const dx = fx - x0, dy = fy - y0;
      const w00 = (1 - dx) * (1 - dy), w10 = dx * (1 - dy);
      const w01 = (1 - dx) * dy, w11 = dx * dy;
      const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4;
      const i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
      const di = (y * w + x) * 4;
      od[di]     = sd[i00] * w00 + sd[i10] * w10 + sd[i01] * w01 + sd[i11] * w11;
      od[di + 1] = sd[i00+1] * w00 + sd[i10+1] * w10 + sd[i01+1] * w01 + sd[i11+1] * w11;
      od[di + 2] = sd[i00+2] * w00 + sd[i10+2] * w10 + sd[i01+2] * w01 + sd[i11+2] * w11;
    }
  }
  outCtx.putImageData(out, 0, 0);
  return outCvs;
}

