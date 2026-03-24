import { $, lsGet, loadImg, warpWithH, squareCropWithMask, canvasToBlob, LS } from "./common.js";

const COLORS = ["#ff3232", "#32c832", "#329aff"];
const KP_ARM = 10;
const KP_RADIUS = 6;
const KP_DOT = 2;

const fusionTile = $("#fusion-tile");
const fusionCvs = $("#fusion-canvas");
const statusEl = $("#status-overlay");
const btnDetect = $("#btn-detect");
const btnDownload = $("#btn-download");
const infoCount = $("#info-count");
const infoTime = $("#info-time");
const kpList = $("#kp-list");

/* ── Fetch snapshot, apply homography, return square-cropped canvas ── */

async function getWarpedSnapshot(camIdx, H) {
  const res = await fetch(`/api/cameras/${camIdx}/snapshot?img_format=png`);
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
    fusionCvs.width = w; fusionCvs.height = h;
  }

  const ctx = fusionCvs.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);

  const n = canvases.length;
  if (!n) return;

  const alpha = 1 / n;
  const sw = canvases[0].width, sh = canvases[0].height;
  const sc = Math.max(w / sw, h / sh);
  const dw = sw * sc, dh = sh * sc;
  const dx = (w - dw) / 2, dy = (h - dh) / 2;
  const cx = dx + dw / 2, cy = dy + dh / 2;
  const r = Math.min(dw, dh) / 2;

  for (const cvs of canvases) {
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.clip();
    ctx.globalAlpha = alpha;
    ctx.drawImage(cvs, dx, dy, dw, dh);
    ctx.restore();
  }
  ctx.globalAlpha = 1;

  if (keypoints?.length) {
    for (let i = 0; i < keypoints.length; i++) {
      const kp = keypoints[i];
      const px = dx + kp.x_norm * dw;
      const py = dy + kp.y_norm * dh;
      const color = COLORS[i % COLORS.length];
      const arm = KP_ARM * dpr;

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1 * dpr;

      ctx.beginPath(); ctx.moveTo(px - arm, py); ctx.lineTo(px + arm, py); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px, py - arm); ctx.lineTo(px, py + arm); ctx.stroke();
      ctx.beginPath(); ctx.arc(px, py, KP_RADIUS * dpr, 0, Math.PI * 2); ctx.stroke();

      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(px, py, KP_DOT * dpr, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }
  }
}

function updateInfo(result) {
  if (!result) { infoCount.textContent = "—"; infoTime.textContent = "—"; kpList.innerHTML = ""; return; }
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

/* ── Download ── */

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
  const assign = lsGet(LS.CAMS, { 1: "", 2: "", 3: "" });
  const storedH = lsGet(LS.H, {});
  for (const s of [1, 2, 3]) {
    if (assign[s] === "") { alert(`Camera ${s} is not assigned. Go to Settings first.`); return; }
  }

  busy = true;
  btnDetect.disabled = true;
  statusEl.classList.add("is-busy");
  updateInfo(null);

  try {
    // USB hub constraint: sequential snapshots
    const canvases = [];
    for (const s of [1, 2, 3]) {
      canvases.push(await getWarpedSnapshot(assign[s], storedH[s] || null));
    }
    renderFusion(canvases, null);

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

btnDetect.addEventListener("click", detect);
btnDownload.addEventListener("click", downloadBlobs);
window.addEventListener("keydown", e => {
  if (e.code === "Space" && !e.repeat && document.activeElement?.tagName !== "BUTTON") {
    e.preventDefault();
    detect();
  }
});
