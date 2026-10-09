"use strict";
const state = {
  players: [],
  squad: null,
  explanation: null,
  meta: null,
  pos: "ALL",
  club: "ALL",
  query: "",
  sortKey: "adjusted_points",
  sortDir: -1,
  errors: {},
  loaded: {},
};
const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const number = (value) =>
  value === null ||
  value === undefined ||
  value === "" ||
  !Number.isFinite(Number(value))
    ? null
    : Number(value);
const fmt = (value, d = 1) =>
  number(value) === null ? "—" : Number(value).toFixed(d);
const pointsOf = (p) => p.adjusted_points ?? p.raw_points ?? null;
const byId = (id) => state.players.find((p) => p.player_id === Number(id));
function safeURL(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}
function opponent(p, short = false) {
  if (!p?.opponent_team) return "—";
  const clubs = {
    Arsenal: "ARS",
    "Aston Villa": "AVL",
    Bournemouth: "BOU",
    Brentford: "BRE",
    Brighton: "BHA",
    Burnley: "BUR",
    Chelsea: "CHE",
    "Coventry City": "COV",
    "Crystal Palace": "CRY",
    Everton: "EVE",
    Fulham: "FUL",
    "Hull City": "HUL",
    Ipswich: "IPS",
    Leeds: "LEE",
    Leicester: "LEI",
    Liverpool: "LIV",
    "Man City": "MCI",
    "Man Utd": "MUN",
    Newcastle: "NEW",
    "Nott'm Forest": "NFO",
    Southampton: "SOU",
    Spurs: "TOT",
    Sunderland: "SUN",
    "West Ham": "WHU",
    Wolves: "WOL",
  };
  const name = short
    ? clubs[p.opponent_team] || p.opponent_team
    : p.opponent_team;
  const venue =
    p.was_home === 1 || p.was_home === true
      ? "H"
      : p.was_home === 0 || p.was_home === false
        ? "A"
        : "";
  return name + (venue ? ` (${venue})` : "");
}
function kitColor(team) {
  return (
    {
      Arsenal: "#b64040",
      Chelsea: "#3f61a6",
      Liverpool: "#b64040",
      Brighton: "#527da7",
      Everton: "#496caf",
      Brentford: "#ba5850",
      Spurs: "#879298",
      "Man City": "#5e9db1",
      "Man Utd": "#b34843",
      Newcastle: "#525e56",
      "Nott'm Forest": "#ba5850",
      "Aston Villa": "#8b5266",
    }[team] || "#627b69"
  );
}
function empty(title, detail = "") {
  return `<div class="empty"><strong>${esc(title)}</strong>${esc(detail)}</div>`;
}
async function loadJSON(key, path) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 12000);
    let res;
    try {
      res = await fetch(path, { cache: "no-store", signal: controller.signal });
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) throw new Error("request failed");
    return await res.json();
  } catch {
    state.errors[key] = true;
    return null;
  }
}
function stat(label, value, detail = "", extra = "") {
  return `<div class="${extra || "summary-stat"}"><div class="stat-label">${esc(label)}</div><div class="stat-value">${esc(value)}</div>${detail ? `<div class="stat-detail">${esc(detail)}</div>` : ""}</div>`;
}
function renderSquad() {
  if (!state.loaded.squad) return;
  const s = state.squad;
  if (!Array.isArray(s?.xi) || !s.xi.length) {
    $("squadSummary").innerHTML = empty(
      "Squad unavailable",
      "The weekly squad is not available yet. Player research is still available.",
    );
    $("pitch").innerHTML = empty(
      "Squad unavailable",
      "Please check back after the next update.",
    );
    $("bench").innerHTML = empty("No bench available");
    return;
  }
  const budget = number(s.total_cost);
  $("squadSummary").innerHTML =
    `<div class="summary-primary"><div class="stat-label">Projected squad points</div><div class="stat-value"><span id="projectedPoints">${fmt(s.predicted_points)}</span><small>pts</small></div><div class="stat-detail">Starting XI · captain multiplier included</div></div>${stat("Squad cost", `£${fmt(s.total_cost)}m`, budget === null ? "" : `£${fmt(100 - budget)}m remaining`)}${stat("Formation", s.formation || "—", "Suggested starting eleven")}${stat("Captain", s.captain?.web_name || "—", s.captain?.team || "")}`;
  $("formationTag").textContent = s.formation || "Starting XI";
  $("pitch").innerHTML = ["GK", "DEF", "MID", "FWD"]
    .map(
      (pos) =>
        `<div class="pitch-row" aria-label="${{ GK: "Goalkeeper", DEF: "Defenders", MID: "Midfielders", FWD: "Forwards" }[pos]}">${s.xi
          .filter((p) => p.position === pos)
          .map(playerCard)
          .join("")}</div>`,
    )
    .join("");
  $("bench").innerHTML = Array.isArray(s.bench)
    ? s.bench.map(playerCard).join("")
    : empty("No bench available");
}
function playerCard(p) {
  const captain = Boolean(
    p.is_captain || p.player_id === state.squad?.captain?.player_id,
  );
  const vice =
    !captain &&
    Boolean(p.is_vice || p.player_id === state.squad?.vice?.player_id);
  const flagged = byId(p.player_id)?.adjustment_reason;
  return `<button class="player-card" data-id="${esc(p.player_id)}" data-captain="${captain}" aria-label="View ${esc(p.web_name)}${captain ? ", captain" : vice ? ", vice-captain" : ""}" title="${esc(p.web_name)} · ${esc(p.team)} · ${esc(opponent(p))}"><span class="kit" style="--kit:${kitColor(p.team)}" aria-hidden="true"></span>${captain || vice ? `<span class="player-tag ${vice ? "vice" : ""}" aria-hidden="true">${captain ? "C" : "V"}</span>` : ""}${flagged ? '<span class="news-indicator" aria-label="News adjustment">!</span>' : ""}<span class="player-name">${esc(p.web_name)}</span><span class="player-opponent">${esc(opponent(p, true))}</span><span class="player-points">${fmt(p.predicted_points)}</span><span class="player-price">£${fmt(p.price)}m</span></button>`;
}
function renderCaptain() {
  if (!state.loaded.squad) return;
  const p = state.squad?.captain,
    v = state.squad?.vice;
  if (!p) {
    $("captainAdvice").innerHTML = empty("Captain advice unavailable");
    return;
  }
  $("captainAdvice").innerHTML =
    `<button class="captain-player" data-id="${esc(p.player_id)}" aria-label="View ${esc(p.web_name)}, captain"><span><span class="captain-name">${esc(p.web_name)}</span><span class="captain-meta">${esc(p.team)} · ${esc(opponent(p))}</span></span><span class="captain-points">${fmt(p.predicted_points)}<small>predicted pts</small></span></button><p class="advice-copy">${esc(state.explanation?.captain_rationale || "The optimizer’s selected captain for this starting eleven. Open player details to explore their predicted points and recent form.")}</p>${v ? `<div class="vice-row"><span>Vice-captain</span><button data-id="${esc(v.player_id)}" aria-label="View ${esc(v.web_name)}, vice-captain">${esc(v.web_name)} <span aria-hidden="true">↗</span></button></div>` : ""}`;
}
function pickMarkup(p) {
  const match = state.players.find((player) => player.web_name === p.player);
  return `<div class="pick-item">${match ? `<button class="pick-title" data-id="${esc(match.player_id)}" aria-label="View ${esc(p.player)}">${esc(p.player)} <span aria-hidden="true">↗</span></button>` : `<h3 class="pick-title">${esc(p.player)}</h3>`}<p>${esc(p.reason)}</p></div>`;
}
function renderPicks() {
  if (!state.loaded.explanation) return;
  const picks = state.explanation?.standout_picks;
  $("standoutPicks").innerHTML =
    Array.isArray(picks) && picks.length
      ? picks.slice(0, 3).map(pickMarkup).join("")
      : empty(
          "Standout picks unavailable",
          "Explore the player predictions for this gameweek.",
        );
}
function renderPlayers() {
  if (!state.loaded.players) return;
  const q = state.query.trim().toLowerCase();
  const rows = state.players.filter(
    (p) =>
      (state.pos === "ALL" || p.position === state.pos) &&
      (state.club === "ALL" || p.team === state.club) &&
      (!q || `${p.web_name || ""} ${p.team || ""}`.toLowerCase().includes(q)),
  );
  rows.sort((a, b) => {
    let av =
        state.sortKey === "adjusted_points" ? pointsOf(a) : a[state.sortKey],
      bv = state.sortKey === "adjusted_points" ? pointsOf(b) : b[state.sortKey];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return (
      state.sortDir *
      (typeof av === "string" ? av.localeCompare(String(bv)) : av - bv)
    );
  });
  $("playerCount").textContent =
    `${rows.length} of ${state.players.length} players`;
  $("playersBody").innerHTML = rows.length
    ? rows
        .map(
          (p) =>
            `<tr data-id="${esc(p.player_id)}"><td><button class="table-player" data-id="${esc(p.player_id)}" aria-label="View ${esc(p.web_name)}">${esc(p.web_name)}</button></td><td>${esc(p.team)}</td><td><span class="pill">${esc(p.position)}</span></td><td>${esc(opponent(p))}</td><td class="num">£${fmt(p.price)}m</td><td class="num">${fmt(p.raw_points, 2)}</td><td class="num predicted-cell">${fmt(pointsOf(p), 2)}</td><td>${p.adjustment_reason ? '<span class="pill flag">Adjusted</span>' : '<span class="muted">—</span>'}</td></tr>`,
        )
        .join("")
    : `<tr><td colspan="8">${empty(state.errors.players ? "Player predictions unavailable" : state.players.length ? "No players match your filters" : "No player predictions yet", state.players.length ? "Try another name or clear your filters." : "Please check back after the next weekly update.")}</td></tr>`;
  document.querySelectorAll("th[data-sort]").forEach((th) => {
    const active = th.dataset.sort === state.sortKey;
    if (active)
      th.setAttribute(
        "aria-sort",
        state.sortDir === 1 ? "ascending" : "descending",
      );
    else th.removeAttribute("aria-sort");
    th.querySelector(".sort-arrow").textContent = active
      ? state.sortDir === 1
        ? "↑"
        : "↓"
      : "";
  });
}
function sourceLabel(source) {
  return source?.startsWith("claude")
    ? `Written from the model output and available news using Claude (${source}).`
    : "Written automatically from the model output.";
}
function renderExplanation() {
  if (!state.loaded.explanation) return;
  const e = state.explanation;
  if (!e) {
    $("explanation").innerHTML = empty(
      "Gameweek insights unavailable",
      "Player predictions and squad information are shown where available.",
    );
    return;
  }
  const picks = Array.isArray(e.standout_picks) ? e.standout_picks : [];
  $("explanation").innerHTML =
    `<article class="panel insight-panel insight-summary"><p class="eyebrow">${esc(e.season || "")} · Gameweek ${esc(e.gameweek ?? "—")}</p><h2>The squad, explained.</h2><p>${esc(e.summary || "No squad summary is available for this gameweek.")}</p><p class="source-note">${esc(sourceLabel(e.source))}</p></article><article class="panel insight-panel"><p class="eyebrow">Captaincy</p><h2>Where the armband goes.</h2><p>${esc(e.captain_rationale || "No captain rationale is available.")}</p>${e.elite_note ? `<div class="detail-section"><h3>The elite-manager signal</h3><p>Elite ownership measures how frequently sampled leading managers select a player. Its influence on this squad is controlled by a tuned weight.</p><p>${esc(e.elite_note)}</p></div>` : ""}</article><article class="panel insight-panel"><p class="eyebrow">Individual recommendations</p><h2>Standout picks</h2>${picks.length ? picks.map(pickMarkup).join("") : empty("No standout picks available")}</article>`;
}
function renderFreshness() {
  const m = state.meta;
  $("seasonLabel").textContent = m?.season
    ? `${m.season} season`
    : "Weekly predictions";
  $("gameweekLabel").textContent =
    m?.gameweek != null
      ? `Gameweek ${m.gameweek} · ${m.season || "FPL"}`
      : "The next gameweek";
  const date = m?.generated_at ? new Date(m.generated_at) : null;
  if (!date || Number.isNaN(date.getTime())) {
    $("generatedAt").textContent = "Update time unavailable";
    $("freshness").textContent = "Weekly snapshot";
    return;
  }
  $("generatedAt").textContent =
    "Updated " +
    new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    }).format(date);
  const stale = Date.now() - date.getTime() > 8 * 24 * 60 * 60 * 1000;
  $("freshness").textContent = stale
    ? "Potentially outdated · more than 8 days old"
    : "Weekly snapshot · not live data";
  $("freshness").classList.toggle("stale", stale);
}
let returnFocus = null,
  returnFocusIdentity = null,
  panelAnimation = null,
  drawerVersion = 0,
  savedScroll = 0;
function motionPanel(open) {
  const panel = document.querySelector(".drawer-panel");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fromTransform = panelAnimation
    ? getComputedStyle(panel).transform
    : null;
  const fromOpacity = panelAnimation ? getComputedStyle(panel).opacity : null;
  panelAnimation?.cancel();
  const mobile = matchMedia("(max-width: 540px)").matches;
  const offset = mobile ? "translateY(100%)" : "translateX(100%)";
  panelAnimation = panel.animate(
    reduced
      ? [{ opacity: fromOpacity || (open ? 0 : 1) }, { opacity: open ? 1 : 0 }]
      : [
          {
            transform: fromTransform || (open ? offset : "none"),
            opacity: fromOpacity || 1,
          },
          { transform: open ? "none" : offset, opacity: 1 },
        ],
    {
      duration: reduced ? 100 : 260,
      easing: "cubic-bezier(.2,.8,.2,1)",
      fill: "both",
    },
  );
  return panelAnimation.finished.catch(() => {});
}
function backgroundInert(value) {
  document
    .querySelectorAll("body > header, body > main, body > footer")
    .forEach((el) => (el.inert = value));
}
function openDrawer(id) {
  const p =
    byId(id) ||
    [...(state.squad?.xi || []), ...(state.squad?.bench || [])].find(
      (p) => p.player_id === Number(id),
    );
  if (!p) return;
  drawerVersion++;
  const wasOpen = !$("drawer").hidden;
  if (!wasOpen) {
    returnFocus = document.activeElement;
    returnFocusIdentity = {
      id: returnFocus?.dataset?.id,
      region: returnFocus?.closest(
        "#pitch,#bench,#captainAdvice,#standoutPicks,#playersBody,#explanation",
      )?.id,
    };
    savedScroll = window.scrollY;
    document.body.style.position = "fixed";
    document.body.style.top = `-${savedScroll}px`;
    document.body.style.width = "100%";
    backgroundInert(true);
  }
  const detailStat = (label, value) =>
    `<div class="detail-stat"><div class="stat-label">${esc(label)}</div><div class="stat-value">${esc(value)}</div></div>`;
  const percent = (value) =>
    number(value) === null ? "—" : `${Math.round(Number(value) * 100)}%`;
  const url = safeURL(p.news_url);
  $("drawerBody").innerHTML =
    `<p class="eyebrow">Player detail</p><h2 id="drawerTitle">${esc(p.web_name)}</h2><p class="dv-sub">${esc(p.team)} · ${esc(p.position)} · £${fmt(p.price)}m</p><div class="detail-projection"><span>Predicted points<br>Next gameweek</span><strong>${fmt(pointsOf(p) ?? p.predicted_points, 2)}</strong></div><h3>The numbers behind the pick</h3><div class="dv-grid">${detailStat("Model points", fmt(p.raw_points, 2))}${detailStat("Opponent", opponent(p))}${detailStat("Recent form", fmt(p.form_ewm, 2))}${detailStat("Avg points · last 5", fmt(p.roll5_total_points, 2))}${detailStat("Avg minutes · last 5", fmt(p.roll5_minutes_played, 0))}${detailStat("Start rate · last 5", percent(p.start_rate_5))}${detailStat("Fixture difficulty", fmt(p.fdr, 1))}${detailStat("Elite ownership", percent(p.elite_template_score))}</div><div class="detail-section"><h3>Team-news context</h3>${p.adjustment_reason ? `<div class="dv-news"><strong>Prediction adjustment ×${fmt(p.adjustment_factor, 2)}</strong><p>${esc(p.adjustment_reason)}</p>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">Read the source ↗</a>` : ""}</div>` : "<p>No news adjustment is recorded for this player. This does not confirm availability.</p>"}</div><div class="detail-section"><h3>Reading these numbers</h3><p>Recent form gives more weight to recent matches. The last-five figures are per-match averages. Start rate is the share of those matches started. Fixture difficulty uses a scale from 1 (easier) to 5 (harder).</p></div><p class="detail-note">Based on the displayed weekly snapshot. A dash means the statistic is unavailable.</p>`;
  $("drawer").hidden = false;
  document.querySelector(".drawer-panel").scrollTop = 0;
  $("drawerClose").focus({ preventScroll: true });
  motionPanel(true);
}
async function closeDrawer() {
  if ($("drawer").hidden) return;
  const version = ++drawerVersion;
  await motionPanel(false);
  if (version !== drawerVersion) return;
  $("drawer").hidden = true;
  panelAnimation?.cancel();
  panelAnimation = null;
  backgroundInert(false);
  document.body.style.position = "";
  document.body.style.top = "";
  document.body.style.width = "";
  window.scrollTo(0, savedScroll);
  const replacement = returnFocusIdentity?.region
    ? [
        ...document
          .getElementById(returnFocusIdentity.region)
          .querySelectorAll("button[data-id]"),
      ].find((el) => el.dataset.id === returnFocusIdentity.id)
    : null;
  const target = returnFocus?.isConnected
    ? returnFocus
    : replacement || document.querySelector(".tab.is-active");
  target?.focus({ preventScroll: true });
}
function activateTab(view, focus = false) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.view === view;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  });
  document
    .querySelectorAll(".view")
    .forEach((panel) => (panel.hidden = panel.id !== `view-${view}`));
}
function initControls() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.view));
    tab.addEventListener("keydown", (event) => {
      const tabs = [...document.querySelectorAll(".tab")],
        index = tabs.indexOf(tab);
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft")
        next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        activateTab(tabs[next].dataset.view, true);
      }
    });
  });
  $("readInsights").addEventListener("click", () => {
    activateTab("insights", true);
    document.querySelector(".navigation").scrollIntoView({ block: "start" });
  });
  $("search").addEventListener("input", (event) => {
    state.query = event.target.value;
    renderPlayers();
  });
  $("clubFilter").addEventListener("change", (event) => {
    state.club = event.target.value;
    renderPlayers();
  });
  document.querySelectorAll("#posFilter .chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      state.pos = chip.dataset.pos;
      updatePosition();
      renderPlayers();
    }),
  );
  $("clearFilters").addEventListener("click", () => {
    state.query = "";
    state.club = "ALL";
    state.pos = "ALL";
    $("search").value = "";
    $("clubFilter").value = "ALL";
    updatePosition();
    renderPlayers();
  });
  document.querySelectorAll("th[data-sort] button").forEach((button) =>
    button.addEventListener("click", () => {
      const key = button.closest("th").dataset.sort;
      state.sortDir =
        state.sortKey === key
          ? -state.sortDir
          : ["web_name", "team", "position", "opponent_team"].includes(key)
            ? 1
            : -1;
      state.sortKey = key;
      renderPlayers();
    }),
  );
  $("main").addEventListener("click", (event) => {
    const target = event.target.closest("button[data-id]");
    if (target) openDrawer(target.dataset.id);
  });
  $("drawerClose").addEventListener("click", closeDrawer);
  $("drawer").addEventListener("click", (event) => {
    if (event.target === $("drawer")) closeDrawer();
  });
  $("drawer").addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closeDrawer();
    }
    if (event.key === "Tab") {
      const targets = [
        ...$("drawer").querySelectorAll("button,a[href]"),
      ].filter((el) => !el.hidden);
      const first = targets[0],
        last = targets.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
}
function updatePosition() {
  document.querySelectorAll("#posFilter .chip").forEach((chip) => {
    const active = chip.dataset.pos === state.pos;
    chip.classList.toggle("is-active", active);
    chip.setAttribute("aria-pressed", String(active));
  });
}
async function main() {
  initControls();
  await Promise.all([
    loadJSON("players", "./data/players.json").then((data) => {
      state.loaded.players = true;
      state.players = Array.isArray(data?.players) ? data.players : [];
      if (!state.meta) state.meta = data?.meta || null;
      const teams = [
        ...new Set(state.players.map((p) => p.team).filter(Boolean)),
      ].sort();
      $("clubFilter").innerHTML =
        '<option value="ALL">All clubs</option>' +
        teams
          .map((team) => `<option value="${esc(team)}">${esc(team)}</option>`)
          .join("");
      renderPlayers();
      renderSquad();
      renderCaptain();
      renderPicks();
      renderExplanation();
      renderFreshness();
    }),
    loadJSON("squad", "./data/squad.json").then((data) => {
      state.loaded.squad = true;
      state.squad = data;
      renderSquad();
      renderCaptain();
    }),
    loadJSON("explanation", "./data/explanation.json").then((data) => {
      state.loaded.explanation = true;
      state.explanation = data;
      renderCaptain();
      renderPicks();
      renderExplanation();
    }),
    loadJSON("meta", "./data/meta.json").then((data) => {
      state.loaded.meta = true;
      if (data) state.meta = data;
      renderFreshness();
    }),
  ]);
}
main();
