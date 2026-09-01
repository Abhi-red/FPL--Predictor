"use strict";

// Static frontend: reads the pipeline's JSON output from ./data/ and renders the
// squad, a searchable players table, and the generated explanation. No build step.

const state = {
  players: [],
  squad: null,
  explanation: null,
  meta: null,
  pos: "ALL",
  query: "",
  sortKey: "adjusted_points",
  sortDir: -1,
};

async function loadJSON(path) {
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch (_) {
    return null;
  }
}

function pointsOf(p) {
  return p.adjusted_points ?? p.raw_points ?? null;
}
function fmt(n, d = 1) {
  return n === null || n === undefined || Number.isNaN(n) ? "–" : Number(n).toFixed(d);
}

// ---- Squad view -----------------------------------------------------------
function renderSquad() {
  const summary = document.getElementById("squadSummary");
  const pitch = document.getElementById("pitch");
  const bench = document.getElementById("bench");
  if (!state.squad || !state.squad.xi) {
    summary.innerHTML = "";
    pitch.innerHTML = '<div class="empty">No squad yet — run the pipeline.</div>';
    bench.innerHTML = "";
    return;
  }
  const s = state.squad;
  summary.innerHTML = [
    ["Formation", s.formation],
    ["Cost", "£" + fmt(s.total_cost, 1) + "m"],
    ["Projected", fmt(s.predicted_points, 1) + " pts"],
    ["Captain", s.captain.web_name],
  ]
    .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");

  const rows = { GK: [], DEF: [], MID: [], FWD: [] };
  s.xi.forEach((p) => rows[p.position].push(p));
  pitch.innerHTML = ["GK", "DEF", "MID", "FWD"]
    .map((pos) => `<div class="pitch-row">${rows[pos].map(playerCard).join("")}</div>`)
    .join("");
  bench.innerHTML = s.bench.map(playerCard).join("");

  document.querySelectorAll(".card").forEach((el) => {
    el.addEventListener("click", () => openDrawer(Number(el.dataset.id)));
  });
}

function playerCard(p) {
  const flagged = byId(p.player_id)?.adjustment_reason ? " flagged" : "";
  let tag = "";
  if (p.is_captain) tag = '<span class="tag">C</span>';
  else if (p.is_vice) tag = '<span class="tag v">V</span>';
  return `<div class="card${flagged}" data-id="${p.player_id}">
    ${tag}
    <div class="name">${p.web_name}</div>
    <div class="meta">${p.team} · £${fmt(p.price, 1)}</div>
    <div class="pts">${fmt(p.predicted_points, 1)}</div>
  </div>`;
}

// ---- Players view --------------------------------------------------------
function byId(id) {
  return state.players.find((p) => p.player_id === id);
}

function renderPlayers() {
  const body = document.getElementById("playersBody");
  const q = state.query.toLowerCase();
  let rows = state.players.filter((p) => {
    if (state.pos !== "ALL" && p.position !== state.pos) return false;
    if (!q) return true;
    return (
      (p.web_name || "").toLowerCase().includes(q) ||
      (p.team || "").toLowerCase().includes(q)
    );
  });
  rows.sort((a, b) => {
    let av = a[state.sortKey];
    let bv = b[state.sortKey];
    if (state.sortKey === "news_flag") {
      av = a.adjustment_reason ? 1 : 0;
      bv = b.adjustment_reason ? 1 : 0;
    }
    if (av === null || av === undefined) av = -Infinity;
    if (bv === null || bv === undefined) bv = -Infinity;
    if (typeof av === "string") return state.sortDir * av.localeCompare(bv);
    return state.sortDir * (av - bv);
  });

  body.innerHTML =
    rows
      .map(
        (p) => `<tr data-id="${p.player_id}">
      <td>${p.web_name}</td>
      <td>${p.team}</td>
      <td><span class="pill">${p.position}</span></td>
      <td class="num">${fmt(p.price, 1)}</td>
      <td class="num">${fmt(p.raw_points, 2)}</td>
      <td class="num">${fmt(pointsOf(p), 2)}</td>
      <td>${p.adjustment_reason ? '<span class="pill flag">flag</span>' : ""}</td>
    </tr>`
      )
      .join("") || '<tr><td colspan="7" class="empty">No players.</td></tr>';

  body.querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.addEventListener("click", () => openDrawer(Number(tr.dataset.id)));
  });
}

// ---- Player drawer -----------------------------------------------------
function openDrawer(id) {
  const p = byId(id);
  if (!p) return;
  const drawer = document.getElementById("drawer");
  const el = document.getElementById("drawerBody");
  const stat = (k, v) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  el.innerHTML = `
    <div class="dv-name">${p.web_name}</div>
    <div class="dv-sub">${p.team} · ${p.position} · £${fmt(p.price, 1)}m</div>
    <div class="dv-grid">
      ${stat("Raw pts", fmt(p.raw_points, 2))}
      ${stat("Adjusted", fmt(pointsOf(p), 2))}
      ${stat("Form (EWM)", fmt(p.form_ewm, 2))}
      ${stat("Pts / last 5", fmt(p.roll5_total_points, 2))}
      ${stat("Mins / last 5", fmt(p.roll5_minutes_played, 0))}
      ${stat("Start rate", p.start_rate_5 == null ? "–" : Math.round(p.start_rate_5 * 100) + "%")}
      ${stat("Fixture diff", fmt(p.fdr, 1))}
      ${stat("Venue", p.was_home == 1 ? "Home" : p.was_home == 0 ? "Away" : "–")}
    </div>
    ${
      p.adjustment_reason
        ? `<div class="dv-news">
             <strong>News adjustment ×${fmt(p.adjustment_factor, 2)}</strong><br/>
             ${p.adjustment_reason}
             ${p.news_url ? `<br/><a href="${p.news_url}" target="_blank" rel="noopener">source</a>` : ""}
           </div>`
        : '<p class="dv-sub">No news flag for this gameweek.</p>'
    }`;
  drawer.hidden = false;
}

// ---- Explanation view --------------------------------------------------
function renderExplanation() {
  const el = document.getElementById("explanation");
  const e = state.explanation;
  if (!e) {
    el.innerHTML = '<div class="empty">No explanation yet — run the pipeline.</div>';
    return;
  }
  const picks = (e.standout_picks || [])
    .map((s) => `<li><strong>${s.player}</strong> — ${s.reason}</li>`)
    .join("");
  el.innerHTML = `
    <h1>Suggested squad — ${e.season} GW${e.gameweek}</h1>
    <p>${e.summary || ""}</p>
    <p><strong>Captaincy.</strong> ${e.captain_rationale || ""}</p>
    <h2>Standout picks</h2>
    <ul>${picks}</ul>
    <p class="dv-sub">Generated by ${e.source || "pipeline"}.</p>`;
}

// ---- Wiring ----------------------------------------------------------
function initTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
      tab.classList.add("is-active");
      document.getElementById("view-" + tab.dataset.view).classList.add("is-active");
    });
  });
}

function initControls() {
  document.getElementById("search").addEventListener("input", (e) => {
    state.query = e.target.value;
    renderPlayers();
  });
  document.querySelectorAll("#posFilter .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll("#posFilter .chip").forEach((c) => c.classList.remove("is-active"));
      chip.classList.add("is-active");
      state.pos = chip.dataset.pos;
      renderPlayers();
    });
  });
  document.querySelectorAll("#playersTable th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      state.sortDir = state.sortKey === key ? -state.sortDir : -1;
      state.sortKey = key;
      renderPlayers();
    });
  });
  document.getElementById("drawerClose").addEventListener("click", () => {
    document.getElementById("drawer").hidden = true;
  });
  document.getElementById("drawer").addEventListener("click", (e) => {
    if (e.target.id === "drawer") e.target.hidden = true;
  });
}

async function main() {
  initTabs();
  initControls();

  const [players, squad, explanation, meta] = await Promise.all([
    loadJSON("./data/players.json"),
    loadJSON("./data/squad.json"),
    loadJSON("./data/explanation.json"),
    loadJSON("./data/meta.json"),
  ]);

  state.players = (players && players.players) || [];
  state.squad = squad;
  state.explanation = explanation;
  state.meta = meta || (players && players.meta) || null;

  if (state.meta) {
    document.getElementById("gwBadge").textContent =
      `${state.meta.season || ""} GW${state.meta.gameweek ?? "–"}`;
    document.getElementById("generatedAt").textContent =
      "Updated " + new Date(state.meta.generated_at).toLocaleString();
  }

  renderSquad();
  renderPlayers();
  renderExplanation();
}

main();
