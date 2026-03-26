import { $, lsGet, loadImg, LS, initSettings, buildCameraBody, buildHomographies } from "./common.js";

await initSettings();

const KP_COLOR = "#fff";
const KP_GLOW = "rgba(255,255,255,0.45)";
const KP_RADIUS = 5;
const AUTO_POLL_MS = 250;

const cameras = lsGet(LS.CAMS, { 1: "", 2: "", 3: "" });

fetch("/api/grabber/start", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ cameras: buildCameraBody() }),
}).catch(e => console.warn("Failed to start grabber:", e));

window.addEventListener("beforeunload", () => {
  navigator.sendBeacon("/api/grabber/stop");
});

/* ── DOM refs ── */

const fusionTile = $("#fusion-tile");
const fusionCvs = $("#fusion-canvas");
const statusEl = $("#status-overlay");
const btnDetect = $("#btn-detect");
const btnDownload = $("#btn-download");
const toggleAuto = $("#toggle-auto");
const infoCount = $("#info-count");
const infoTime = $("#info-time");
const kpList = $("#kp-list");

/* ── Render fusion blend with keypoints ── */

function renderFusion(imgs, keypoints) {
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

  const n = imgs.length;
  if (!n) return;

  const alpha = 1 / n;
  const sw = imgs[0].naturalWidth, sh = imgs[0].naturalHeight;
  const sc = Math.max(w / sw, h / sh);
  const dw = sw * sc, dh = sh * sc;
  const dx = (w - dw) / 2, dy = (h - dh) / 2;
  const cx = dx + dw / 2, cy = dy + dh / 2;
  const r = Math.min(dw, dh) / 2;

  for (const img of imgs) {
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.clip();
    ctx.globalAlpha = alpha;
    ctx.drawImage(img, dx, dy, dw, dh);
    ctx.restore();
  }
  ctx.globalAlpha = 1;

  if (keypoints?.length) {
    for (let i = 0; i < keypoints.length; i++) {
      const kp = keypoints[i];
      const px = dx + kp.x_norm * dw;
      const py = dy + kp.y_norm * dh;
      const kr = KP_RADIUS * dpr;

      ctx.save();
      ctx.shadowColor = KP_GLOW;
      ctx.shadowBlur = 10 * dpr;
      ctx.fillStyle = KP_COLOR;
      ctx.beginPath();
      ctx.arc(px, py, kr, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      ctx.strokeStyle = "rgba(0,0,0,0.4)";
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.arc(px, py, kr, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

/* ── Display result ── */

let lastB64Images = null;

async function displayResult(result) {
  const imgs = await Promise.all(
    result.images.map(b64 => loadImg("data:image/jpeg;base64," + b64))
  );
  lastB64Images = result.images;
  btnDownload.disabled = false;
  renderFusion(imgs, result.keypoints);
  updateInfo(result);
}

function updateInfo(result) {
  if (!result) { infoCount.textContent = "—"; infoTime.textContent = "—"; kpList.innerHTML = ""; return; }
  infoCount.textContent = result.count;
  infoTime.textContent = `${result.time_ms} ms (total ${result.total_ms} ms)`;
  kpList.innerHTML = "";
  for (let i = 0; i < result.keypoints.length; i++) {
    const kp = result.keypoints[i];
    const row = document.createElement("div");
    row.className = "kp-row";
    row.innerHTML =
      `<span><span class="kp-dot"></span>Dart ${kp.dart}</span>` +
      `<span class="kp-coords">(${kp.x_norm.toFixed(3)}, ${kp.y_norm.toFixed(3)}) conf ${kp.confidence.toFixed(3)}</span>`;
    kpList.appendChild(row);
  }
}

function downloadImages() {
  if (!lastB64Images) return;
  for (let i = 0; i < lastB64Images.length; i++) {
    const a = document.createElement("a");
    a.href = "data:image/jpeg;base64," + lastB64Images[i];
    a.download = `cam${i + 1}.jpg`;
    a.click();
  }
}

/* ── Manual detect ── */

let busy = false;

async function detect() {
  if (busy) return;
  for (const s of [1, 2, 3]) {
    if (cameras[s] === "") { alert(`Camera ${s} is not assigned. Go to Settings first.`); return; }
  }

  busy = true;
  btnDetect.disabled = true;
  statusEl.classList.add("is-busy");
  updateInfo(null);

  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cameras: buildCameraBody(), homographies: buildHomographies() }),
    });
    if (!res.ok) throw new Error(`Detection failed: ${res.status}`);
    await displayResult(await res.json());
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

/* ── Auto-detect: poll for results from background change detection ── */

let autoPolling = null;
let lastResultId = 0;

function startAutoPolling() {
  if (autoPolling) return;
  lastResultId = 0;
  autoPolling = setInterval(pollAutoResult, AUTO_POLL_MS);
}

function stopAutoPolling() {
  if (autoPolling) { clearInterval(autoPolling); autoPolling = null; }
}

async function pollAutoResult() {
  try {
    const res = await fetch(`/api/grabber/result?after=${lastResultId}`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.result && data.id > lastResultId) {
      lastResultId = data.id;
      await displayResult(data.result);
    }
  } catch { /* ignore poll errors */ }
}

async function setAutoDetect(enabled) {
  try {
    await fetch("/api/grabber/auto", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled, homographies: buildHomographies() }),
    });
    if (enabled) {
      startAutoPolling();
    } else {
      stopAutoPolling();
    }
  } catch (e) {
    console.warn("Failed to toggle auto-detect:", e);
    toggleAuto.checked = false;
  }
}

/* ── Event bindings ── */

btnDetect.addEventListener("click", detect);
btnDownload.addEventListener("click", downloadImages);
toggleAuto.addEventListener("change", () => setAutoDetect(toggleAuto.checked));

window.addEventListener("keydown", e => {
  if (e.code === "Space" && !e.repeat && document.activeElement?.tagName !== "BUTTON") {
    e.preventDefault();
    detect();
  }
});
