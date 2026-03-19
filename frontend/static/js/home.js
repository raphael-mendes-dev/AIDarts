(function () {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  const modeList = $("#mode-list");
  const playerSelector = $("#player-selector");
  const startLink = $("#start-link");
  if (!modeList || !playerSelector || !startLink) return;

  const allCards = () => $$(".mode-card[data-mode]", modeList);
  const enabledCards = () => allCards().filter(c => c.getAttribute("aria-disabled") !== "true");
  const allPlayers = () => $$(".player-btn", playerSelector);

  const selected = {
    card: () => $(".mode-card.is-selected", modeList),
    player: () => $(".player-btn.is-selected", playerSelector),
  };

  let rafId = 0;

  function measureConfig(card) {
    const cfg = $(".mode-config", card);
    if (cfg) card.style.setProperty("--config-h", cfg.scrollHeight + 12 + "px");
  }

  function updateFrame() {
    const frame = $(".player-frame", playerSelector);
    const btn = selected.player();
    if (!frame || !btn) return;
    const base = playerSelector.getBoundingClientRect();
    const r = btn.getBoundingClientRect();
    playerSelector.style.setProperty("--pf-w", r.width + "px");
    playerSelector.style.setProperty("--pf-h", r.height + "px");
    playerSelector.style.setProperty("--pf-x", r.left - base.left + "px");
    playerSelector.style.setProperty("--pf-y", r.top - base.top + "px");
  }

  function trackLayout(ms) {
    cancelAnimationFrame(rafId);
    const t0 = performance.now();
    const tick = now => {
      allCards().forEach(measureConfig);
      updateFrame();
      rafId = now - t0 < ms ? requestAnimationFrame(tick) : 0;
    };
    rafId = requestAnimationFrame(tick);
  }

  function updateLink() {
    const card = selected.card();
    const btn = selected.player();
    if (!card || !btn) return;
    const mode = card.dataset.mode;
    const checkin = ($(`input[name="checkin-${mode}"]:checked`) || {}).value || "straight";
    const checkout = ($(`input[name="checkout-${mode}"]:checked`) || {}).value || "straight";
    const params = new URLSearchParams({
      mode,
      players: btn.value,
      check_in: checkin === "straight" ? "straight_in" : checkin + "_in",
      check_out: checkout === "straight" ? "straight_out" : checkout + "_out",
    });
    startLink.href = "/game?" + params;
  }

  function selectMode(card) {
    if (!card || card.getAttribute("aria-disabled") === "true") return;
    allCards().forEach(c => {
      const on = c === card;
      c.classList.toggle("is-selected", on);
      c.setAttribute("aria-expanded", on);
    });
    updateLink();
    trackLayout(420);
  }

  function selectPlayer(btn) {
    if (!btn) return;
    allPlayers().forEach(b => {
      const on = b === btn;
      b.classList.toggle("is-selected", on);
      b.setAttribute("aria-checked", on);
    });
    updateLink();
    trackLayout(320);
  }

  allCards().forEach(card => {
    card.addEventListener("click", () => selectMode(card));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectMode(card); }
    });
  });

  allPlayers().forEach(btn => btn.addEventListener("click", () => selectPlayer(btn)));
  modeList.addEventListener("change", e => { if (e.target.type === "radio") updateLink(); });
  window.addEventListener("resize", () => trackLayout(260));

  if (document.fonts?.ready) document.fonts.ready.then(() => trackLayout(200)).catch(() => {});

  if (!selected.card()) selectMode(enabledCards()[0]);
  if (!selected.player()) selectPlayer(allPlayers()[0]);
  allCards().forEach(measureConfig);
  updateLink();
  trackLayout(420);
})();
