import { $, lsGet, LS, initSettings, buildCameraBody, buildHomographies } from "./common.js";

await initSettings();

const CONFIDENCE_THRESHOLD = 0.2;
const DART_COLOR = "#fff";
const DART_GLOW = "rgba(255,255,255,0.45)";
const DART_RADIUS = 5;
const HIT_RADIUS = 25;
const DRAG_THRESHOLD = 8;
const AUTO_POLL_MS = 250;

/* ── Parse game config from URL ── */

const params = new URLSearchParams(window.location.search);
const config = {
  mode: params.get("mode") || "x01",
  target: parseInt(params.get("target") || "501", 10),
  checkout: params.get("checkout") || "single",
  players: parseInt(params.get("players") || "1", 10),
};

/* ── Game state ── */

const state = {
  scores: [],           // remaining score per player
  currentPlayer: 0,
  turnDarts: [],        // [{label, score, multiplier, segment, x_norm, y_norm}, ...]
  turnStartScore: 0,    // score at start of current turn (for bust revert)
  gameOver: false,
  busted: false,
  waitingForClear: false, // true until board is confirmed empty at game start
  history: [],          // [{player, darts: [{label, score}...], total, remaining, busted}]
};

/* ── Client-side scorer (mirrors backend/scorer.py) ── */

const _R = 225.5;
const RINGS = { bull: 6.35/_R, outerBull: 15.9/_R, triIn: 99/_R, triOut: 107/_R, dblIn: 162/_R, dblOut: 170/_R };
const SECTORS = [20,1,18,4,13,6,10,15,2,17,3,19,7,16,8,11,14,9,12,5];

// Mirrors backend/scorer.py — kept client-side for instant UI feedback without a round-trip.
// If scoring geometry changes, update both.
function scoreDart(xn, yn) {
  const dx = (xn - 0.5) * 2, dy = (yn - 0.5) * 2;
  const r = Math.hypot(dx, dy);
  if (r <= RINGS.bull) return { segment: 25, multiplier: 2, score: 50, label: "Bull" };
  if (r <= RINGS.outerBull) return { segment: 25, multiplier: 1, score: 25, label: "Outer Bull" };
  if (r > RINGS.dblOut) return { segment: 0, multiplier: 0, score: 0, label: "Miss" };
  let a = ((Math.atan2(dx, -dy) + Math.PI / 20) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
  const seg = SECTORS[Math.floor(a / (Math.PI / 10)) % 20];
  const mul = r > RINGS.dblIn ? 2 : (r > RINGS.triIn && r <= RINGS.triOut) ? 3 : 1;
  return { segment: seg, multiplier: mul, score: seg * mul, label: mul === 3 ? `T${seg}` : mul === 2 ? `D${seg}` : `${seg}` };
}

function initGame() {
  state.scores = Array(config.players).fill(config.target);
  state.currentPlayer = 0;
  state.turnDarts = [];
  state.turnStartScore = config.target;
  state.gameOver = false;
  state.busted = false;
  state.waitingForClear = true;
  state.history = [];
  renderScoreboard();
  renderTurn();
  renderBoard();
  renderHistory();
  setStatus("Checking board...");
  $("#win-overlay").hidden = true;
  clearOverlay.hidden = true;
  boardWrap.classList.add("is-waiting");
  btnNext.disabled = true;
  btnMiss.disabled = true;
}

/* ── DOM refs ── */

const scoreboardEl = $("#scoreboard");
const turnDartsEl = $("#turn-darts");
const turnTotalEl = $("#turn-total");
const statusEl = $("#status-text");
const boardCanvas = $("#board-canvas");
const boardWrap = $(".board-wrap");
const btnMiss = $("#btn-miss");
const btnNext = $("#btn-next");
const historyEl = $("#history");
const clearOverlay = $("#clear-overlay");

/* ── Render scoreboard ── */

function renderScoreboard() {
  scoreboardEl.innerHTML = "";
  for (let i = 0; i < config.players; i++) {
    const card = document.createElement("div");
    const active = i === state.currentPlayer;
    card.className = "player-card" + (active ? " is-active" : "") + (active && state.busted ? " is-bust" : "");
    card.innerHTML =
      `<span class="player-name">P${i + 1}</span>` +
      `<span class="player-score">${state.scores[i]}</span>`;
    scoreboardEl.appendChild(card);
  }
}

/* ── Render current turn darts ── */

function renderTurn() {
  for (let i = 0; i < 3; i++) {
    const slot = $(`#dart-${i + 1}`);
    const dart = state.turnDarts[i];
    if (dart) {
      slot.textContent = dart.label;
      slot.className = "dart-slot " + (dart.score === 0 ? "is-miss" : "is-scored");
    } else {
      slot.textContent = "—";
      slot.className = "dart-slot";
    }
  }
  const total = state.turnDarts.reduce((s, d) => s + d.score, 0);
  turnTotalEl.textContent = total;
}

/* ── Render dart markers on board reference image ── */

function renderBoard() {
  const wrap = boardCanvas.parentElement;
  const sz = wrap.clientWidth;
  if (boardCanvas.width !== sz || boardCanvas.height !== sz) {
    boardCanvas.width = sz;
    boardCanvas.height = sz;
  }
  const ctx = boardCanvas.getContext("2d");
  ctx.clearRect(0, 0, sz, sz);

  for (let i = 0; i < state.turnDarts.length; i++) {
    const d = state.turnDarts[i];
    const x = d.x_norm * sz;
    const y = d.y_norm * sz;

    ctx.save();
    ctx.shadowColor = DART_GLOW;
    ctx.shadowBlur = 10;
    ctx.fillStyle = DART_COLOR;
    ctx.beginPath();
    ctx.arc(x, y, DART_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.strokeStyle = "rgba(0,0,0,0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, DART_RADIUS, 0, Math.PI * 2);
    ctx.stroke();
  }
}

/* ── Render history ── */

function renderHistory() {
  historyEl.innerHTML = "";
  if (state.history.length === 0) return;

  const table = document.createElement("table");
  table.className = "history-table";
  table.innerHTML = `<thead><tr>
    <th>#</th><th>Player</th><th>D1</th><th>D2</th><th>D3</th><th>Turn</th><th>Left</th>
  </tr></thead>`;
  const tbody = document.createElement("tbody");

  for (let i = 0; i < state.history.length; i++) {
    const h = state.history[i];
    const tr = document.createElement("tr");
    if (h.busted) tr.className = "is-bust";

    const darts = [0, 1, 2].map(j => h.darts[j]
      ? `<td class="${h.darts[j].score === 0 ? 'is-miss' : ''}">${h.darts[j].label}</td>`
      : "<td>—</td>"
    ).join("");

    tr.innerHTML =
      `<td>${i + 1}</td>` +
      `<td>P${h.player + 1}</td>` +
      darts +
      `<td class="turn-col">${h.busted ? "BUST" : h.total}</td>` +
      `<td>${h.remaining}</td>`;
    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  historyEl.appendChild(table);
  historyEl.scrollTop = historyEl.scrollHeight;
}

/* ── Score application ── */

function applyDart(dart) {
  if (state.turnDarts.length >= 3 || state.gameOver || state.busted || state.waitingForClear) return;

  state.turnDarts.push(dart);
  const pIdx = state.currentPlayer;
  const newScore = state.scores[pIdx] - dart.score;

  if (newScore < 0 || (config.checkout === "double" && newScore === 1)) {
    bust();
    return;
  }

  if (newScore === 0 && config.checkout === "double" && dart.multiplier !== 2) {
    bust();
    return;
  }

  state.scores[pIdx] = newScore;
  renderScoreboard();
  renderTurn();
  renderBoard();
  if (newScore === 0) {
    win(pIdx);
    return;
  }

  if (state.turnDarts.length >= 3) {
    setStatus("Remove darts from board...");
    btnNext.disabled = false;
  } else {
    setStatus(`Dart ${state.turnDarts.length + 1} of 3 — throw next dart`);
  }
}

function recordTurn(busted = false) {
  state.history.push({
    player: state.currentPlayer,
    darts: state.turnDarts.map(d => ({ label: d.label, score: d.score })),
    total: busted ? 0 : state.turnDarts.reduce((s, d) => s + d.score, 0),
    remaining: state.scores[state.currentPlayer],
    busted,
  });
  renderHistory();
}

function bust() {
  state.busted = true;
  state.scores[state.currentPlayer] = state.turnStartScore;
  setStatus("BUST — correct the score or remove darts");
  renderScoreboard();
  renderTurn();
  renderBoard();
  btnNext.disabled = false;
}

function win(playerIdx) {
  recordTurn();
  state.gameOver = true;
  $("#win-message").textContent = `Player ${playerIdx + 1} wins!`;
  $("#win-overlay").hidden = false;
  stopAutoDetect();
}

function nextPlayer() {
  if (state.turnDarts.length > 0) recordTurn(state.busted);
  state.currentPlayer = (state.currentPlayer + 1) % config.players;
  state.turnDarts = [];
  state.turnStartScore = state.scores[state.currentPlayer];
  state.busted = false;
  renderScoreboard();
  renderTurn();
  renderBoard();
  btnNext.disabled = true;
  setStatus("Throw your darts...");
  // Reset auto-detect baseline for new turn
  fetch("/api/grabber/baseline", { method: "POST" }).catch(() => {});
}

function recordMiss() {
  applyDart({ label: "Miss", score: 0, multiplier: 0, segment: 0, x_norm: -1, y_norm: -1 });
}

function setStatus(text) {
  statusEl.textContent = text;
}

/* ── Recalculate turn from current turnDarts (for manual edits) ── */

function recalcTurn() {
  state.scores[state.currentPlayer] = state.turnStartScore;
  state.busted = false;

  for (const d of state.turnDarts) {
    const ns = state.scores[state.currentPlayer] - d.score;
    if (ns < 0 || (config.checkout === "double" && ns === 1) ||
        (ns === 0 && config.checkout === "double" && d.multiplier !== 2)) {
      state.scores[state.currentPlayer] = state.turnStartScore;
      state.busted = true;
      break;
    }
    state.scores[state.currentPlayer] = ns;
    if (ns === 0) { win(state.currentPlayer); return; }
  }

  renderScoreboard();
  renderTurn();
  renderBoard();

  if (state.busted) {
    setStatus("BUST — correct the score or remove darts");
    btnNext.disabled = false;
  } else if (state.turnDarts.length >= 3) {
    setStatus("Remove darts from board...");
    btnNext.disabled = false;
  } else {
    setStatus(state.turnDarts.length
      ? `Dart ${state.turnDarts.length + 1} of 3 — throw next dart`
      : "Throw your darts...");
    btnNext.disabled = true;
  }
  fetch("/api/grabber/baseline", { method: "POST" }).catch(() => {});
}

/* ── Board interaction: tap to place/remove, drag to move ── */

let _drag = null; // { idx, startX, startY, moved }

function boardCoords(e) {
  const r = boardCanvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
}

function hitTest(nx, ny, rect) {
  const thresh = HIT_RADIUS / rect.width;
  for (let i = 0; i < state.turnDarts.length; i++) {
    const d = state.turnDarts[i];
    if (d.x_norm < 0) continue; // miss (off-board)
    if (Math.hypot(d.x_norm - nx, d.y_norm - ny) < thresh) return i;
  }
  return -1;
}

boardCanvas.addEventListener("pointerdown", e => {
  if (state.gameOver || state.waitingForClear) return;
  const rect = boardCanvas.getBoundingClientRect();
  const p = { x: (e.clientX - rect.left) / rect.width, y: (e.clientY - rect.top) / rect.height };
  const hit = hitTest(p.x, p.y, rect);

  if (hit >= 0) {
    _drag = { idx: hit, startX: e.clientX, startY: e.clientY, moved: false };
    boardCanvas.setPointerCapture(e.pointerId);
  } else if (state.turnDarts.length < 3 && !state.busted) {
    const info = scoreDart(p.x, p.y);
    applyDart({ ...info, x_norm: p.x, y_norm: p.y });
  }
});

boardCanvas.addEventListener("pointermove", e => {
  if (!_drag) return;
  if (!_drag.moved && Math.hypot(e.clientX - _drag.startX, e.clientY - _drag.startY) < DRAG_THRESHOLD) return;
  _drag.moved = true;
  const p = boardCoords(e);
  state.turnDarts[_drag.idx].x_norm = p.x;
  state.turnDarts[_drag.idx].y_norm = p.y;
  renderBoard();
});

boardCanvas.addEventListener("pointerup", e => {
  if (!_drag) return;
  const { idx, moved } = _drag;
  _drag = null;

  if (moved) {
    const d = state.turnDarts[idx];
    Object.assign(d, scoreDart(d.x_norm, d.y_norm));
  } else {
    state.turnDarts.splice(idx, 1);
  }
  recalcTurn();
});

/* ── Auto-detect integration ── */

// Start grabber
fetch("/api/grabber/start", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ cameras: buildCameraBody() }),
}).then(() => {
  // Enable auto-detect
  return fetch("/api/grabber/auto", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: true, homographies: buildHomographies() }),
  });
}).catch(e => console.warn("Failed to start grabber:", e));

window.addEventListener("beforeunload", () => {
  navigator.sendBeacon("/api/grabber/stop");
});

/* ── Poll for auto-detect results ── */

let lastResultId = 0;
let pollInterval = null;

function startAutoDetect() {
  if (pollInterval) return;
  pollInterval = setInterval(pollResult, AUTO_POLL_MS);
}

function stopAutoDetect() {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

async function pollResult() {
  if (state.gameOver) return;

  try {
    const res = await fetch(`/api/grabber/result?after=${lastResultId}`);
    if (!res.ok) return;
    const data = await res.json();
    if (!data.result || data.id <= lastResultId) return;
    lastResultId = data.id;

    handleDetectionResult(data.result);
  } catch { /* ignore poll errors */ }
}

function handleDetectionResult(result) {
  const darts = result.keypoints.filter(kp => kp.confidence >= CONFIDENCE_THRESHOLD);

  // Waiting for empty board at game start
  if (state.waitingForClear) {
    if (darts.length === 0) {
      state.waitingForClear = false;
      boardWrap.classList.remove("is-waiting");
      clearOverlay.hidden = true;
      setStatus("Throw your darts...");
      btnMiss.disabled = false;
      fetch("/api/grabber/baseline", { method: "POST" }).catch(() => {});
    } else {
      clearOverlay.hidden = false;
    }
    return;
  }

  // Board cleared (0 darts) after turn complete or bust → advance
  if (darts.length === 0 && (state.turnDarts.length >= 3 || state.busted)) {
    nextPlayer();
    return;
  }

  // Busted: ignore new detections, only removal (above) matters
  if (state.busted) return;

  // Waiting for board clear after 3 darts (still darts on board)
  if (state.turnDarts.length >= 3) return;

  if (darts.length === 0) return;

  state.turnDarts = darts.map(kp => ({
    label: kp.label, score: kp.score,
    multiplier: kp.multiplier, segment: kp.segment,
    x_norm: kp.x_norm, y_norm: kp.y_norm,
  }));
  recalcTurn();
}

/* ── Event bindings ── */

btnMiss.addEventListener("click", recordMiss);
btnNext.addEventListener("click", nextPlayer);
$("#btn-restart").addEventListener("click", () => {
  initGame();
  lastResultId = 0;
  startAutoDetect();
  fetch("/api/grabber/baseline", { method: "POST" }).catch(() => {});
});

/* ── Game title ── */

$("#game-title").textContent = `${config.target} — ${config.checkout === "double" ? "Double Out" : "Single Out"}`;

/* ── Start ── */

initGame();
startAutoDetect();
