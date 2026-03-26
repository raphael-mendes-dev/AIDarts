import { $, $$, lsGet, lsSet, loadImg, warpWithH, EPS, LS, initSettings } from "./common.js";

await initSettings();

const HANDLE_KEYS = ["nw", "ne", "se", "sw"];
const WIRE_SRC = {
  nw: [0.23346, 0.23346],
  ne: [0.76654, 0.23346],
  se: [0.76654, 0.76654],
  sw: [0.23346, 0.76654],
};
const ROTATION_STEP_DEG = 18;
const ANCHOR_MOVE_THRESHOLD = 0.002;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function isFinite2(p) { return Array.isArray(p) && p.length === 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]); }

function cloneAnchors(a) {
  if (!a) return null;
  const o = {};
  for (const k of HANDLE_KEYS) { if (!isFinite2(a[k])) return null; o[k] = [a[k][0], a[k][1]]; }
  return o;
}

function warpToDataUrl(img, H) {
  return warpWithH(img, H).toDataURL("image/png");
}

/* ── Linear algebra ── */

function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let best = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[best][c])) best = r;
    [M[c], M[best]] = [M[best], M[c]];
    if (Math.abs(M[c][c]) < EPS) return null;
    const p = M[c][c];
    for (let j = c; j <= n; j++) M[c][j] /= p;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map(r => r[n]);
}

function normH(H) {
  if (!H) return null;
  const s = H[2][2];
  if (Math.abs(s) < EPS) return null;
  const out = H.map(r => r.map(v => v / s));
  return out.flat().every(Number.isFinite) ? out : null;
}

function mulH(A, B) {
  const R = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    R[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j];
  return R;
}

function h4pt(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [sx, sy] = src[i], [dx, dy] = dst[i];
    A.push([sx, sy, 1, 0, 0, 0, -dx*sx, -dx*sy]);
    A.push([0, 0, 0, sx, sy, 1, -dy*sx, -dy*sy]);
    b.push(dx); b.push(dy);
  }
  const h = solve(A, b);
  if (!h) return null;
  return normH([[h[0],h[1],h[2]], [h[3],h[4],h[5]], [h[6],h[7],1]]);
}

function hToPixels(H, sz) {
  return normH([
    [H[0][0], H[0][1], H[0][2]*sz],
    [H[1][0], H[1][1], H[1][2]*sz],
    [H[2][0]/sz, H[2][1]/sz, H[2][2]],
  ]);
}

function hToCss(H) {
  const s = H[2][2];
  if (Math.abs(s) < EPS) return null;
  const [a,b,c] = H[0].map(v=>v/s), [d,e,f] = H[1].map(v=>v/s), [g,h] = [H[2][0]/s, H[2][1]/s];
  const vals = [a,d,0,g, b,e,0,h, 0,0,1,0, c,f,0,1];
  return vals.every(Number.isFinite) ? `matrix3d(${vals.join(",")})` : null;
}

/* ── State ── */

let detected  = lsGet(LS.DET, []);
let assign    = lsGet(LS.CAMS, { 1: "", 2: "", 3: "" });
let calibSave = lsGet(LS.CAL, {});
let storedH   = lsGet(LS.H, {});
let fusionVis = { 1: true, 2: true, 3: true };
let previews  = {};
let originals = {};
let fusionVer = 0;

const modal = {
  open: false, slot: null, anchors: null,
  dragging: "", pointerId: null, handleEl: null,
};

/* ── DOM refs ── */

const btnScan    = $("#btn-scan");
const scanStatus = $("#scan-status");
const selects    = [1,2,3].map(i => $(`#cam-sel-${i}`));
const thumbEls   = [1,2,3].map(i => $(`#cam-thumb-${i}`));
const modalEl    = $("#calib-modal");
const modalTitle = $("#modal-title");
const modalTile  = $("#modal-tile");
const modalImg   = $("#modal-img");
const wireLayer  = $("#wire-layer");
const wireSvg    = $("#wire-svg");
const fusionTile = $("#fusion-tile");
const fusionCvs  = $("#fusion-canvas");

const wireHandleEls = {};
$$("[data-handle]", wireLayer).forEach(el => { wireHandleEls[el.dataset.handle] = el; });

/* ── Helpers ── */

function revokePreview(slot) {
  if (previews[slot]?.url) URL.revokeObjectURL(previews[slot].url);
}
function revokeOriginal(slot) {
  if (originals[slot]?.url) URL.revokeObjectURL(originals[slot].url);
}

/* ── 1. Cameras ── */

if (detected.length) { populateSelects(); scanStatus.textContent = `${detected.length} camera(s)`; }

btnScan.addEventListener("click", async () => {
  btnScan.disabled = true; scanStatus.textContent = "Scanning…";
  try {
    detected = (await (await fetch("/api/cameras")).json()).cameras;
    lsSet(LS.DET, detected);
    scanStatus.textContent = detected.length ? `${detected.length} camera(s)` : "None found";
    populateSelects();
  } catch (e) {
    console.warn("Camera scan failed:", e);
    scanStatus.textContent = "Scan failed";
  }
  btnScan.disabled = false;
});

function populateSelects() {
  selects.forEach((sel, i) => {
    const slot = i + 1;
    sel.innerHTML = '<option value="">— none —</option>';
    detected.forEach(c => { const o = document.createElement("option"); o.value = c.index; o.textContent = `${c.name} (${c.width}×${c.height})`; sel.appendChild(o); });
    sel.disabled = !detected.length;
    if (assign[slot] !== "") sel.value = assign[slot];
  });
}

selects.forEach((sel, i) => {
  sel.addEventListener("change", () => { assign[i+1] = sel.value; lsSet(LS.CAMS, assign); if (sel.value) loadPreview(i+1); else clearPreview(i+1); });
});

async function loadPreview(slot) {
  const idx = assign[slot]; if (idx === "") return;
  const tile = $(`#tile-${slot}`); tile.classList.add("is-busy");
  try {
    const r = await fetch(`/api/cameras/${idx}/snapshot`);
    if (!r.ok) return;
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const img = await loadImg(url);
    tile.classList.remove("is-busy"); tile.classList.add("is-live");
    revokeOriginal(slot);
    originals[slot] = { url, img };
    if (storedH[slot]) {
      await new Promise(resolve => applyTotalH(slot, resolve));
    } else {
      setPreviewEverywhere(slot, url, img);
      scheduleFusion();
    }
  } catch (e) {
    console.warn("Preview load failed:", e);
    tile.classList.remove("is-busy");
  }
}

// Warp the ORIGINAL image with the cumulative H — never warp an already-warped image
function applyTotalH(slot, cb) {
  const orig = originals[slot];
  const H = storedH[slot];
  if (!orig || !H) { if (cb) cb(); return; }
  const warpedUrl = warpToDataUrl(orig.img, H);
  loadImg(warpedUrl).then(warpedImg => {
    setPreviewEverywhere(slot, warpedUrl, warpedImg);
    scheduleFusion();
    if (cb) cb();
  });
}

function setPreviewEverywhere(slot, url, img) {
  revokePreview(slot);
  previews[slot] = { url, img };
  $(`#img-${slot}`).src = url;
  $(`#tile-${slot}`).classList.add("is-live");
  setThumb(slot, url);
}

function clearPreview(slot) {
  $(`#tile-${slot}`).classList.remove("is-live", "is-busy");
  $(`#img-${slot}`).removeAttribute("src");
  thumbEls[slot-1].innerHTML = "";
  revokePreview(slot);
  revokeOriginal(slot);
  delete previews[slot];
  delete originals[slot];
  scheduleFusion();
}

function setThumb(slot, url) {
  const t = thumbEls[slot-1]; t.innerHTML = "";
  const i = document.createElement("img"); i.src = url; t.appendChild(i);
}

// USB cameras on a hub fail if opened concurrently
(async () => {
  for (const s of [1, 2, 3]) {
    if (assign[s] !== "") await loadPreview(s);
  }
})();

/* ── 3. Calibration Modal ── */

function getLayout() {
  const tR = modalTile.getBoundingClientRect();
  const tw = tR.width, th = tR.height;
  if (tw < 2 || th < 2) return null;
  const nw = modalImg.naturalWidth || tw, nh = modalImg.naturalHeight || th;
  if (nw < 2 || nh < 2) return null;
  const sc = Math.min(tw / nw, th / nh);
  const iw = nw * sc, ih = nh * sc;
  const il = (tw - iw) / 2, it = (th - ih) / 2;
  const osz = Math.min(iw, ih);
  const ol = il + (iw - osz) / 2, ot = it + (ih - osz) / 2;
  return { iw, ih, il, it, osz, ol, ot };
}

function overlayToImage(p, L) {
  return [(L.ol - L.il + p[0] * L.osz) / L.iw, (L.ot - L.it + p[1] * L.osz) / L.ih];
}

function imageToOverlay(p, L) {
  return [(p[0] * L.iw - (L.ol - L.il)) / L.osz, (p[1] * L.ih - (L.ot - L.it)) / L.osz];
}

function defaultAnchors(L) {
  const a = {};
  for (const k of HANDLE_KEYS) a[k] = overlayToImage(WIRE_SRC[k], L);
  return a;
}

function ensureAnchors(L) {
  if (modal.anchors) return true;
  const saved = cloneAnchors(calibSave[modal.slot]);
  if (saved) { modal.anchors = saved; return true; }
  modal.anchors = defaultAnchors(L);
  return true;
}

function buildOverlayH(anchors) {
  return h4pt(HANDLE_KEYS.map(k => WIRE_SRC[k]), HANDLE_KEYS.map(k => anchors[k]));
}

/* Wire overlay render (RAF-throttled during drag) */

let wireRafPending = false;

function scheduleRenderWire() {
  if (wireRafPending) return;
  wireRafPending = true;
  requestAnimationFrame(() => { wireRafPending = false; renderWire(); });
}

function renderWire() {
  if (!modal.open) return;
  const L = getLayout();
  if (!L || !ensureAnchors(L)) return;

  wireLayer.style.left   = L.ol + "px";
  wireLayer.style.top    = L.ot + "px";
  wireLayer.style.width  = L.osz + "px";
  wireLayer.style.height = L.osz + "px";
  wireLayer.style.transform = "none";
  wireLayer.classList.toggle("is-dragging", Boolean(modal.dragging));

  const H = buildOverlayH(modal.anchors);
  if (H) {
    const imgToLayer = [
      [L.iw / L.osz, 0, -(L.ol - L.il) / L.osz],
      [0, L.ih / L.osz, -(L.ot - L.it) / L.osz],
      [0, 0, 1],
    ];
    const overlayToLayer = normH(mulH(imgToLayer, H));
    const px = overlayToLayer ? hToPixels(overlayToLayer, L.osz) : null;
    wireSvg.style.transform = px ? (hToCss(px) || "") : "";
  } else {
    wireSvg.style.transform = "";
  }

  for (const k of HANDLE_KEYS) {
    const el = wireHandleEls[k];
    const lp = imageToOverlay(modal.anchors[k], L);
    el.style.left = (lp[0] * 100) + "%";
    el.style.top  = (lp[1] * 100) + "%";
    el.style.display = "";
    el.classList.toggle("is-active", modal.dragging === k);
  }
}

/* Open / close */

$$("[data-open-calib]").forEach(btn => {
  btn.addEventListener("click", () => {
    const slot = Number(btn.dataset.openCalib);
    if (!previews[slot]) return;
    openModal(slot);
  });
});

function openModal(slot) {
  modal.open = true;
  modal.slot = slot;
  modal.anchors = cloneAnchors(calibSave[slot]);
  modal.dragging = "";

  modalTitle.textContent = `Calibration — CAM ${slot}`;
  modalImg.src = previews[slot].url;
  modalTile.classList.add("is-live");
  modalEl.classList.add("is-open");
  modalEl.setAttribute("aria-hidden", "false");

  requestAnimationFrame(renderWire);
}

function closeModal() {
  endDrag(null, true);
  modal.open = false;
  modal.slot = null;
  modal.anchors = null;
  modalEl.classList.remove("is-open");
  modalEl.setAttribute("aria-hidden", "true");
  modalTile.classList.remove("is-live");
  modalImg.removeAttribute("src");
  wireSvg.style.transform = "";
  wireLayer.style.cssText = "";
  wireLayer.classList.remove("is-dragging");
  for (const k of HANDLE_KEYS) { wireHandleEls[k].style.left = ""; wireHandleEls[k].style.top = ""; wireHandleEls[k].classList.remove("is-active"); }
  scheduleFusion();
}

$("#modal-backdrop").addEventListener("click", closeModal);
$("#btn-close").addEventListener("click", closeModal);
window.addEventListener("keydown", e => { if (e.key === "Escape" && modal.open) closeModal(); });

$("#btn-reset").addEventListener("click", async () => {
  const slot = modal.slot;
  if (!slot) return;

  delete storedH[slot];
  lsSet(LS.H, storedH);
  delete calibSave[slot];
  lsSet(LS.CAL, calibSave);

  const idx = assign[slot];
  if (idx !== "") {
    const tile = $(`#tile-${slot}`);
    tile.classList.add("is-busy");
    try {
      const r = await fetch(`/api/cameras/${idx}/snapshot`);
      if (!r.ok) return;
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const img = await loadImg(url);
      tile.classList.remove("is-busy");
      revokeOriginal(slot);
      originals[slot] = { url, img };
      setPreviewEverywhere(slot, url, img);
      if (modal.open && modal.slot === slot) {
        modalImg.src = url;
        modalTile.classList.add("is-live");
        const L = getLayout();
        if (L) modal.anchors = defaultAnchors(L);
        renderWire();
      }
      scheduleFusion();
    } catch (e) {
      console.warn("Reset snapshot failed:", e);
      tile.classList.remove("is-busy");
    }
  } else {
    const L = getLayout();
    if (L) modal.anchors = defaultAnchors(L);
    renderWire();
  }
});

modalImg.addEventListener("load", () => { if (modal.open) requestAnimationFrame(renderWire); });

/* Rotation — norm→pixel→center→rotate→uncenter→norm */

function buildRotationH(degrees, img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const rad = degrees * Math.PI / 180;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  return normH([
    [cos,          -sin * h / w,  0.5 - cos * 0.5 + sin * h / (2 * w)],
    [sin * w / h,   cos,          0.5 - sin * w / (2 * h) - cos * 0.5],
    [0,             0,            1],
  ]);
}

function applyRotation(degrees) {
  const slot = modal.slot;
  if (!slot || !originals[slot] || modal.dragging) return;
  const Hrot = buildRotationH(degrees, originals[slot].img);
  if (!Hrot) return;
  const prev = storedH[slot] || [[1,0,0],[0,1,0],[0,0,1]];
  const Htotal = normH(mulH(prev, Hrot));
  if (!Htotal) return;
  storedH[slot] = Htotal;
  lsSet(LS.H, storedH);
  applyTotalH(slot, () => {
    modalImg.src = previews[slot].url;
    const L = getLayout();
    if (L) modal.anchors = defaultAnchors(L);
    calibSave[slot] = cloneAnchors(modal.anchors);
    lsSet(LS.CAL, calibSave);
    renderWire();
  });
}

$("#btn-rotate-left").addEventListener("click", () => applyRotation(ROTATION_STEP_DEG));
$("#btn-rotate-right").addEventListener("click", () => applyRotation(-ROTATION_STEP_DEG));

/* Handle dragging */

for (const key of HANDLE_KEYS) {
  const el = wireHandleEls[key];
  el.addEventListener("pointerdown", e => {
    if (!modal.open) return;
    e.preventDefault();
    e.stopPropagation();
    const L = getLayout();
    if (!L || !ensureAnchors(L)) return;
    modal.dragging = key;
    modal.pointerId = e.pointerId;
    modal.handleEl = el;
    try { el.setPointerCapture(e.pointerId); } catch {}
    applyPointer(e, L);
    renderWire();
  });
}

window.addEventListener("pointermove", e => {
  if (!modal.dragging || e.pointerId !== modal.pointerId) return;
  e.preventDefault();
  const L = getLayout();
  if (!L) return;
  applyPointer(e, L);
  scheduleRenderWire();
});

window.addEventListener("pointerup",     e => endDrag(e, false));
window.addEventListener("pointercancel", e => endDrag(e, true));
window.addEventListener("blur", () => endDrag(null, true));

function applyPointer(e, L) {
  const layerR = wireLayer.getBoundingClientRect();
  if (layerR.width < 1) return;
  const lx = clamp((e.clientX - layerR.left) / layerR.width, 0, 1);
  const ly = clamp((e.clientY - layerR.top) / layerR.height, 0, 1);
  const imgPt = overlayToImage([lx, ly], L);
  const newPt = [clamp(imgPt[0], 0, 1), clamp(imgPt[1], 0, 1)];
  const prev = modal.anchors[modal.dragging];
  modal.anchors[modal.dragging] = newPt;
  if (!buildOverlayH(modal.anchors)) {
    modal.anchors[modal.dragging] = prev;
  }
}

function endDrag(e, cancelled) {
  if (!modal.dragging) return;
  if (e && e.pointerId !== modal.pointerId) return;
  if (!cancelled) {
    const L = getLayout();
    if (L && e) applyPointer(e, L);
  }
  if (modal.handleEl) { try { modal.handleEl.releasePointerCapture(modal.pointerId); } catch {} }
  modal.dragging = "";
  modal.pointerId = null;
  modal.handleEl = null;
  renderWire();
  if (!cancelled && modal.slot && modal.anchors) applyWarp();
}

function applyWarp() {
  const slot = modal.slot;
  const L = getLayout();
  if (!L || !modal.anchors) return;
  const defaults = defaultAnchors(L);

  let moved = false;
  for (const k of HANDLE_KEYS) {
    if (Math.abs(modal.anchors[k][0] - defaults[k][0]) > ANCHOR_MOVE_THRESHOLD ||
        Math.abs(modal.anchors[k][1] - defaults[k][1]) > ANCHOR_MOVE_THRESHOLD) { moved = true; break; }
  }
  if (!moved) return;

  const Hinc = h4pt(HANDLE_KEYS.map(k => defaults[k]), HANDLE_KEYS.map(k => modal.anchors[k]));
  if (!Hinc) return;
  const prev = storedH[slot] || [[1,0,0],[0,1,0],[0,0,1]];
  const Htotal = normH(mulH(prev, Hinc));
  if (!Htotal) return;

  storedH[slot] = Htotal;
  lsSet(LS.H, storedH);
  applyTotalH(slot, () => {
    modalImg.src = previews[slot].url;
    modal.anchors = defaultAnchors(getLayout() || L);
    calibSave[slot] = cloneAnchors(modal.anchors);
    lsSet(LS.CAL, calibSave);
    renderWire();
  });
}

window.addEventListener("resize", () => { if (modal.open) renderWire(); scheduleFusion(); });

/* ── 4. Fusion Preview ── */

$$("[data-fusion]").forEach(input => {
  const s = Number(input.dataset.fusion);
  input.addEventListener("change", () => { fusionVis[s] = input.checked; scheduleFusion(); });
});

let fusionPending = false;
function scheduleFusion() {
  if (fusionPending) return;
  fusionPending = true;
  requestAnimationFrame(() => { fusionPending = false; renderFusion(); });
}

function renderFusion() {
  const ver = ++fusionVer;
  const srcs = [1,2,3].filter(s => fusionVis[s] && previews[s]);
  if (!srcs.length) { fusionTile.classList.remove("is-live"); return; }

  const imgs = srcs.map(s => previews[s].img);
  if (ver !== fusionVer) return;

  const cw = Math.max(1, fusionTile.clientWidth), ch = Math.max(1, fusionTile.clientHeight);
  const dpr = Math.max(1, devicePixelRatio || 1);
  const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
  if (fusionCvs.width !== w || fusionCvs.height !== h) { fusionCvs.width = w; fusionCvs.height = h; }

  const ctx = fusionCvs.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, w, h);

  const alpha = 1 / imgs.length;
  for (const img of imgs) {
    const sw = img.naturalWidth || w, sh = img.naturalHeight || h;
    const sc = Math.max(w / sw, h / sh);
    const dw = sw * sc, dh = sh * sc;
    const dx = (w - dw) / 2, dy = (h - dh) / 2;
    const cx = dx + dw / 2, cy = dy + dh / 2;
    const r = Math.min(dw, dh) / 2;
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.clip();
    ctx.globalAlpha = alpha;
    ctx.drawImage(img, dx, dy, dw, dh);
    ctx.restore();
  }
  ctx.globalAlpha = 1;
  fusionTile.classList.add("is-live");
}
