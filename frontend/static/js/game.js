import { $ } from "./common.js";

const params = new URLSearchParams(window.location.search);

const config = {
  mode: params.get("mode") || "x01",
  target: parseInt(params.get("target") || "501", 10),
  checkout: params.get("checkout") || "single",
  players: parseInt(params.get("players") || "1", 10),
};

$("#game-title").textContent = `${config.target} — ${config.checkout === "double" ? "Double Out" : "Single Out"}`;
$("#game-info").textContent = `${config.players} player${config.players > 1 ? "s" : ""} — Game starting...`;
