import { $, $$, lsGet, loadImg, LS } from "./common.js";

const CONFIDENCE_THRESHOLD = 0.2;
const DART_COLORS = ["#ff3232", "#32c832", "#329aff"];
const DART_MARKER_RADIUS = 8;
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
  previousDartCount: 0, // track dart count changes for one-by-one detection
};

function initGame() {
  state.scores = Array(config.players).fill(config.target);
  state.currentPlayer = 0;
  state.turnDarts = [];
  state.turnStartScore = config.target;
  state.gameOver = false;
  state.previousDartCount = 0;
  renderScoreboard();
  renderTurn();
  renderBoard();
  setStatus("Throw your darts...");
  $("#win-overlay").hidden = true;
  btnUndo.disabled = true;
  btnNext.disabled = true;
}

/* ── DOM refs ── */

const scoreboardEl = $("#scoreboard");
const turnDartsEl = $("#turn-darts");
const turnTotalEl = $("#turn-total");
const statusEl = $("#status-text");
const boardCanvas = $("#board-canvas");
const btnUndo = $("#btn-undo");
const btnMiss = $("#btn-miss");
const btnNext = $("#btn-next");

/* ── Render scoreboard ── */

function renderScoreboard() {
  scoreboardEl.innerHTML = "";
  for (let i = 0; i < config.players; i++) {
    const card = document.createElement("div");
    card.className = "player-card" + (i === state.currentPlayer ? " is-active" : "");
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
    const color = DART_COLORS[i % DART_COLORS.length];

    // Crosshair + circle marker
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x - DART_MARKER_RADIUS, y); ctx.lineTo(x + DART_MARKER_RADIUS, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y - DART_MARKER_RADIUS); ctx.lineTo(x, y + DART_MARKER_RADIUS); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, DART_MARKER_RADIUS, 0, Math.PI * 2); ctx.stroke();

    // Center dot
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();

    // Label
    ctx.font = "bold 14px sans-serif";
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 3;
    const lbl = d.label;
    const tx = x + DART_MARKER_RADIUS + 4;
    const ty = y - DART_MARKER_RADIUS;
    ctx.strokeText(lbl, tx, ty);
    ctx.fillText(lbl, tx, ty);
  }
}

/* ── Score application ── */

function applyDart(dart) {
  if (state.turnDarts.length >= 3 || state.gameOver) return;

  state.turnDarts.push(dart);
  const pIdx = state.currentPlayer;
  const newScore = state.scores[pIdx] - dart.score;

  // Bust check
  if (newScore < 0 || (config.checkout === "double" && newScore === 1)) {
    bust();
    return;
  }

  // Double out: must finish on a double (or bull)
  if (newScore === 0 && config.checkout === "double" && dart.multiplier !== 2) {
    bust();
    return;
  }

  state.scores[pIdx] = newScore;
  renderScoreboard();
  renderTurn();
  renderBoard();
  btnUndo.disabled = false;

  // Win check
  if (newScore === 0) {
    win(pIdx);
    return;
  }

  // After 3 darts, prompt to wait for board clear
  if (state.turnDarts.length >= 3) {
    setStatus("Remove darts from board...");
    btnNext.disabled = false;
  } else {
    setStatus(`Dart ${state.turnDarts.length + 1} of 3 — throw next dart`);
  }
}

function bust() {
  state.scores[state.currentPlayer] = state.turnStartScore;
  setStatus("BUST! Score reverted.");
  highlightBust();
  renderScoreboard();
  renderTurn();
  renderBoard();
  btnUndo.disabled = true;

  // Auto-advance after bust
  setTimeout(() => nextPlayer(), 2000);
}

function highlightBust() {
  const cards = $$(".player-card", scoreboardEl);
  if (cards[state.currentPlayer]) {
    cards[state.currentPlayer].classList.add("is-bust");
    setTimeout(() => cards[state.currentPlayer]?.classList.remove("is-bust"), 2000);
  }
}

function win(playerIdx) {
  state.gameOver = true;
  $("#win-message").textContent = `Player ${playerIdx + 1} wins!`;
  $("#win-overlay").hidden = false;
  stopAutoDetect();
}

function nextPlayer() {
  state.currentPlayer = (state.currentPlayer + 1) % config.players;
  state.turnDarts = [];
  state.turnStartScore = state.scores[state.currentPlayer];
  state.previousDartCount = 0;
  renderScoreboard();
  renderTurn();
  renderBoard();
  btnUndo.disabled = true;
  btnNext.disabled = true;
  setStatus("Throw your darts...");
  // Reset auto-detect baseline for new turn
  fetch("/api/grabber/baseline", { method: "POST" }).catch(() => {});
}

function undoLastDart() {
  if (state.turnDarts.length === 0) return;
  const dart = state.turnDarts.pop();
  state.scores[state.currentPlayer] += dart.score;
  state.previousDartCount = state.turnDarts.length;
  renderScoreboard();
  renderTurn();
  renderBoard();
  btnUndo.disabled = state.turnDarts.length === 0;
  btnNext.disabled = true;
  setStatus(`Dart ${state.turnDarts.length + 1} of 3 — throw next dart`);
  // Reset baseline so auto-detect re-evaluates from current state
  fetch("/api/grabber/baseline", { method: "POST" }).catch(() => {});
}

function recordMiss() {
  applyDart({ label: "Miss", score: 0, multiplier: 0, segment: 0, x_norm: -1, y_norm: -1 });
}

function setStatus(text) {
  statusEl.textContent = text;
}

/* ── Auto-detect integration ── */

const cameras = lsGet(LS.CAMS, { 1: "", 2: "", 3: "" });
const homographies = lsGet(LS.H, {});

function buildCameraBody() {
  const body = {};
  for (const s of ["1", "2", "3"]) {
    if (cameras[s] !== "") body[s] = Number(cameras[s]);
  }
  return body;
}

function buildHomographies() {
  const h = {};
  for (const s of ["1", "2", "3"]) {
    if (homographies[s]) h[s] = homographies[s];
  }
  return h;
}

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
  // Filter out low-confidence detections
  const darts = result.keypoints.filter(kp => kp.confidence >= CONFIDENCE_THRESHOLD);
  const dartCount = darts.length;

  // Waiting for board clear after 3 darts
  if (state.turnDarts.length >= 3) {
    if (dartCount === 0) {
      // Board is clear — advance to next player
      nextPlayer();
    }
    return;
  }

  // One-by-one detection: only process if dart count increased
  if (dartCount > state.previousDartCount) {
    // New dart(s) appeared — process only the new ones
    const newDarts = darts.slice(state.previousDartCount);
    for (const kp of newDarts) {
      if (state.turnDarts.length >= 3) break;
      applyDart({
        label: kp.label,
        score: kp.score,
        multiplier: kp.multiplier,
        segment: kp.segment,
        x_norm: kp.x_norm,
        y_norm: kp.y_norm,
      });
    }
    state.previousDartCount = dartCount;
  }
}

/* ── Event bindings ── */

btnUndo.addEventListener("click", undoLastDart);
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
