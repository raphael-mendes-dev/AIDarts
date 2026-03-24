(function () {
  "use strict";

  const EPS = 1e-12;
  const LS_CAMS = "aidarts_cam_assign";
  const LS_H = "aidarts_homography";
  const COLORS = ["#ff3232", "#32c832", "#329aff"];

  const $ = (s, p) => (p || document).querySelector(s);

  function lsGet(k, fb) { try { return JSON.parse(localStorage.getItem(k)) || fb; } catch { return fb; } }

  /* ── DOM refs ── */

  const fusionTile = $("#fusion-tile");
  const fusionCvs = $("#fusion-canvas");
  const statusEl = $("#status-overlay");
  const btnDetect = $("#btn-detect");
  const btnDownload = $("#btn-download");
  const infoCount = $("#info-count");
  const infoTime = $("#info-time");
  const kpList = $("#kp-list");

  /* ── Homography warp (same as settings.js) ── */

  function warpWithH(img, H) {
    const w = img.naturalWidth, h = img.naturalHeight;
    const srcCvs = document.createElement("canvas");
    srcCvs.width = w; srcCvs.height = h;
    const srcCtx = srcCvs.getContext("2d");
    srcCtx.drawImage(img, 0, 0);
    const sd = srcCtx.getImageData(0, 0, w, h).data;
    const outCvs = document.createElement("canvas");
    outCvs.width = w; outCvs.height = h;
    const outCtx = outCvs.getContext("2d");
    // Pre-fill with solid black (alpha=255) so unmapped pixels are opaque black
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
        od[di] = sd[i00] * w00 + sd[i10] * w10 + sd[i01] * w01 + sd[i11] * w11;
        od[di + 1] = sd[i00 + 1] * w00 + sd[i10 + 1] * w10 + sd[i01 + 1] * w01 + sd[i11 + 1] * w11;
        od[di + 2] = sd[i00 + 2] * w00 + sd[i10 + 2] * w10 + sd[i01 + 2] * w01 + sd[i11 + 2] * w11;
      }
    }
    outCtx.putImageData(out, 0, 0);
    return outCvs;
  }

  /* ── Load image from URL as promise ── */

  function loadImg(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }

  /* ── Canvas to Blob as promise ── */

  function canvasToBlob(cvs) {
    return new Promise(resolve => cvs.toBlob(resolve, "image/png"));
  }

  /* ── Crop to centered square with circular board mask ── */

  function squareCropWithMask(srcCvs) {
    const w = srcCvs.width, h = srcCvs.height;
    const sz = Math.min(w, h);
    const sx = (w - sz) / 2, sy = (h - sz) / 2;
    const out = document.createElement("canvas");
    out.width = sz; out.height = sz;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, sz, sz);
    // Circular clip — radius = sz/2, matching training data mask
    ctx.save();
    ctx.beginPath();
    ctx.arc(sz / 2, sz / 2, sz / 2, 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(srcCvs, sx, sy, sz, sz, 0, 0, sz, sz);
    ctx.restore();
    return out;
  }

  /* ── Fetch snapshot, apply homography, return square canvas ── */

  async function getWarpedSnapshot(camIdx, H) {
    const res = await fetch(`/api/cameras/${camIdx}/snapshot?format=png`);
    if (!res.ok) throw new Error(`Snapshot failed for camera ${camIdx}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    try {
      const img = await loadImg(url);
      let cvs;
      if (H) {
        cvs = warpWithH(img, H);
      } else {
        cvs = document.createElement("canvas");
        cvs.width = img.naturalWidth;
        cvs.height = img.naturalHeight;
        cvs.getContext("2d").drawImage(img, 0, 0);
      }
      return squareCropWithMask(cvs);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  /* ── Render fusion blend with keypoints ── */

  function renderFusion(canvases, keypoints) {
    const cw = Math.max(1, fusionTile.clientWidth);
    const ch = Math.max(1, fusionTile.clientHeight);
    const dpr = Math.max(1, devicePixelRatio || 1);
    const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
    if (fusionCvs.width !== w || fusionCvs.height !== h) {
      fusionCvs.width = w;
      fusionCvs.height = h;
    }

    const ctx = fusionCvs.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);

    const n = canvases.length;
    if (!n) return;

    const alpha = 1 / n;

    // Compute draw rect fitting all canvases (use first for reference)
    const sw = canvases[0].width, sh = canvases[0].height;
    const sc = Math.max(w / sw, h / sh);
    const dw = sw * sc, dh = sh * sc;
    const dx = (w - dw) / 2, dy = (h - dh) / 2;
    const cx = dx + dw / 2, cy = dy + dh / 2;
    const r = Math.min(dw, dh) / 2;

    for (const cvs of canvases) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.clip();
      ctx.globalAlpha = alpha;
      ctx.drawImage(cvs, dx, dy, dw, dh);
      ctx.restore();
    }
    ctx.globalAlpha = 1;

    // Draw keypoints — small and thin crosshairs
    if (keypoints && keypoints.length) {
      for (let i = 0; i < keypoints.length; i++) {
        const kp = keypoints[i];
        const px = dx + kp.x_norm * dw;
        const py = dy + kp.y_norm * dh;
        const color = COLORS[i % COLORS.length];
        const arm = 10 * dpr;
        const dotR = 2 * dpr;

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1 * dpr;

        // Crosshair
        ctx.beginPath();
        ctx.moveTo(px - arm, py);
        ctx.lineTo(px + arm, py);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(px, py - arm);
        ctx.lineTo(px, py + arm);
        ctx.stroke();

        // Small circle
        ctx.beginPath();
        ctx.arc(px, py, 6 * dpr, 0, Math.PI * 2);
        ctx.stroke();

        // Center dot
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, dotR, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
      }
    }
  }

  /* ── Update info panel ── */

  function updateInfo(result) {
    if (!result) {
      infoCount.textContent = "—";
      infoTime.textContent = "—";
      kpList.innerHTML = "";
      return;
    }

    infoCount.textContent = result.count;
    infoTime.textContent = result.time_ms + " ms";

    kpList.innerHTML = "";
    for (let i = 0; i < result.keypoints.length; i++) {
      const kp = result.keypoints[i];
      const color = COLORS[i % COLORS.length];
      const row = document.createElement("div");
      row.className = "kp-row";
      row.innerHTML =
        `<span><span class="kp-dot" style="background:${color}"></span>Dart ${kp.dart}</span>` +
        `<span class="kp-coords">(${kp.x_norm.toFixed(3)}, ${kp.y_norm.toFixed(3)}) conf ${kp.confidence.toFixed(3)}</span>`;
      kpList.appendChild(row);
    }
  }

  /* ── Download helper ── */

  let lastBlobs = null;

  function downloadBlobs() {
    if (!lastBlobs) return;
    for (let i = 0; i < lastBlobs.length; i++) {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(lastBlobs[i]);
      a.download = `cam${i + 1}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
    }
  }

  /* ── Detect flow ── */

  let busy = false;

  async function detect() {
    if (busy) return;

    const assign = lsGet(LS_CAMS, { 1: "", 2: "", 3: "" });
    const storedH = lsGet(LS_H, {});

    // Check that all 3 cameras are assigned
    const slots = [1, 2, 3];
    for (const s of slots) {
      if (assign[s] === "") {
        alert(`Camera ${s} is not assigned. Go to Settings first.`);
        return;
      }
    }

    busy = true;
    btnDetect.disabled = true;
    statusEl.classList.add("is-busy");
    updateInfo(null);

    try {
      // Take snapshots sequentially (USB hub constraint) and apply homography
      const canvases = [];
      for (const s of slots) {
        const H = storedH[s] || null;
        const cvs = await getWarpedSnapshot(assign[s], H);
        canvases.push(cvs);
      }

      // Render fusion immediately (before detection) so user sees the images
      renderFusion(canvases, null);

      // Send warped images to backend
      const blobs = await Promise.all(canvases.map(canvasToBlob));
      lastBlobs = blobs;
      btnDownload.disabled = false;
      const form = new FormData();
      form.append("cam1", blobs[0], "cam1.png");
      form.append("cam2", blobs[1], "cam2.png");
      form.append("cam3", blobs[2], "cam3.png");

      const res = await fetch("/api/detect", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Detection failed: ${res.status}`);
      const result = await res.json();

      // Redraw fusion with keypoints
      renderFusion(canvases, result.keypoints);
      updateInfo(result);
    } catch (err) {
      console.error("Detection error:", err);
      infoCount.textContent = "Error";
      infoTime.textContent = "—";
    } finally {
      busy = false;
      btnDetect.disabled = false;
      statusEl.classList.remove("is-busy");
    }
  }

  /* ── Event bindings ── */

  btnDetect.addEventListener("click", detect);
  btnDownload.addEventListener("click", downloadBlobs);

  window.addEventListener("keydown", e => {
    if (e.code === "Space" && !e.repeat && document.activeElement?.tagName !== "BUTTON") {
      e.preventDefault();
      detect();
    }
  });
})();
