/* Ryder Cup 2027 Predictor — dashboard app
 * Loads docs/data/model_output.json (produced by model/pipeline.py) and renders
 * the dashboard. The match engine below is a JavaScript port of
 * model/matchplay.py, running on the same per-hole distributions, so numbers
 * match the precomputed matrices.
 */
"use strict";

let DATA = null;
let SHARP = 1.9;
const REPO_URL = ""; // set after publishing, e.g. https://github.com/<user>/ryder-cup-2027
const MATCH_N = 25000;

/* ===================== engine (port of model/matchplay.py) ===================== */
const BASE = { eagle: 0.012, birdie: 0.230, par: 0.570, bogey: 0.155, dbl: 0.033 };
const LAMBDA_SENS = 2.45;

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

function holeDist(rating) {
  const lam = clamp(-rating / (18 * LAMBDA_SENS), -0.18, 0.14);
  const e = clamp(BASE.eagle + 0.30 * lam, 0, 0.09);
  const b = clamp(BASE.birdie + lam, 0.02, 0.46);
  const g = clamp(BASE.bogey - 0.55 * lam, 0.015, 0.34);
  const d = clamp(BASE.dbl - 0.15 * lam, 0.003, 0.14);
  let p = Math.max(1 - e - b - g - d, 0.05);
  const s = e + b + p + g + d;
  return [e / s, b / s, p / s, g / s, d / s];
}

function distOf(p) {
  const h = p.hole_dist;
  if (!h) return holeDist(0);
  return [h.eagle, h.birdie, h.par, h.bogey, h.dbl];
}

function foursomesDistOf(pa, pb) {
  const r = (pa.rating + pb.rating) / 2;
  return holeDist(r * SHARP + 0.12);
}

function drawHoles(dist, n, holes) {
  const c0 = dist[0], c1 = c0 + dist[1], c2 = c1 + dist[2], c3 = c2 + dist[3];
  const out = new Float32Array(n * holes);
  for (let i = 0; i < n * holes; i++) {
    const u = Math.random();
    out[i] = u < c0 ? -2 : u < c1 ? -1 : u < c2 ? 0 : u < c3 ? 1 : 2;
  }
  return out;
}

function scoreLabel(margin, remaining) {
  return remaining <= 0 ? "1UP" : margin + "&" + remaining;
}

function extraLabel(hole, aWins) {
  const h = hole >= 21 ? "21st+" : (18 + hole) + "th";
  return h + (aWins ? " (A)" : " (B)");
}

/* Simulate n 18-hole matches between distA (A = left/USA perspective) and distB. */
function simSingles(distA, distB, n) {
  n = n || MATCH_N;
  const H = 18;
  const sa = drawHoles(distA, n, H);
  const sb = drawHoles(distB, n, H);
  const win = {}, lose = {};
  let nWin = 0, nLose = 0, nExtra = 0, nHalved = 0;
  for (let i = 0; i < n; i++) {
    let run = 0, ended = false, holeEnd = 0, margin = 0;
    for (let h = 0; h < H; h++) {
      const d = Math.sign(sb[i * H + h] - sa[i * H + h]);
      run += d;
      const remaining = H - (h + 1);
      if (Math.abs(run) > remaining) { ended = true; holeEnd = h + 1; margin = run; break; }
    }
    if (ended) {
      const lab = scoreLabel(Math.abs(margin), H - holeEnd);
      if (margin > 0) { nWin++; win[lab] = (win[lab] || 0) + 1; }
      else { nLose++; lose[lab] = (lose[lab] || 0) + 1; }
    } else {
      nExtra++;
      // sudden-death extra holes
      const c0a = distA[0], c1a = c0a + distA[1], c2a = c1a + distA[2], c3a = c2a + distA[3];
      const c0b = distB[0], c1b = c0b + distB[1], c2b = c1b + distB[2], c3b = c2b + distB[3];
      let decided = false;
      for (let x = 1; x <= 10 && !decided; x++) {
        const ua = Math.random(), ub = Math.random();
        const A = ua < c0a ? -2 : ua < c1a ? -1 : ua < c2a ? 0 : ua < c3a ? 1 : 2;
        const B = ub < c0b ? -2 : ub < c1b ? -1 : ub < c2b ? 0 : ub < c3b ? 1 : 2;
        if (A < B) { nWin++; decided = true; const lab = extraLabel(x, true); win[lab] = (win[lab] || 0) + 1; }
        else if (A > B) { nLose++; decided = true; const lab = extraLabel(x, false); lose[lab] = (lose[lab] || 0) + 1; }
      }
      if (!decided) { nHalved++; nWin += 0.5; }
    }
  }
  return { n, nWin, nLose, nExtra, nHalved, win, lose };
}

/* Four-ball: best ball of each pair. */
function simFourball(dA1, dA2, dB1, dB2, n) {
  n = n || MATCH_N;
  const H = 18;
  const a1 = drawHoles(dA1, n, H), a2 = drawHoles(dA2, n, H);
  const b1 = drawHoles(dB1, n, H), b2 = drawHoles(dB2, n, H);
  const win = {}, lose = {};
  let nWin = 0, nLose = 0, nExtra = 0, nHalved = 0;
  for (let i = 0; i < n; i++) {
    let run = 0, ended = false, holeEnd = 0, margin = 0;
    for (let h = 0; h < H; h++) {
      const o = i * H + h;
      const pa = Math.min(a1[o], a2[o]);
      const pb = Math.min(b1[o], b2[o]);
      const d = Math.sign(pb - pa);
      run += d;
      const remaining = H - (h + 1);
      if (Math.abs(run) > remaining) { ended = true; holeEnd = h + 1; margin = run; break; }
    }
    if (ended) {
      const lab = scoreLabel(Math.abs(margin), H - holeEnd);
      if (margin > 0) { nWin++; win[lab] = (win[lab] || 0) + 1; }
      else { nLose++; lose[lab] = (lose[lab] || 0) + 1; }
    } else {
      nExtra++;
      let decided = false;
      for (let x = 1; x <= 10 && !decided; x++) {
        const pick = (arr) => { const u = Math.random(); return u < arr[0] ? -2 : u < arr[0] + arr[1] ? -1 : u < arr[0] + arr[1] + arr[2] ? 0 : u < arr[0] + arr[1] + arr[2] + arr[3] ? 1 : 2; };
        const pa = Math.min(pick(dA1), pick(dA2));
        const pb = Math.min(pick(dB1), pick(dB2));
        if (pa < pb) { nWin++; decided = true; const lab = extraLabel(x, true); win[lab] = (win[lab] || 0) + 1; }
        else if (pa > pb) { nLose++; decided = true; const lab = extraLabel(x, false); lose[lab] = (lose[lab] || 0) + 1; }
      }
      if (!decided) { nWin += 0.5; nHalved++; }
    }
  }
  return { n, nWin, nLose, nExtra, nHalved, win, lose };
}

/* ===================== helpers ===================== */
const $ = (id) => document.getElementById(id);
const pct = (x, d) => (100 * x).toFixed(d === undefined ? 1 : d) + "%";

const FLAGS = {
  "USA": "🇺🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
  "Northern Ireland": "🍀", "Ireland": "🇮🇪", "Spain": "🇪🇸", "France": "🇫🇷", "Germany": "🇩🇪",
  "Sweden": "🇸🇪", "Denmark": "🇩🇰", "Norway": "🇳🇴", "Finland": "🇫🇮", "Austria": "🇦🇹",
  "Belgium": "🇧🇪", "Netherlands": "🇳🇱", "Italy": "🇮🇹", "New Zealand": "🇳🇿", "Canada": "🇨🇦",
  "Australia": "🇦🇺", "South Africa": "🇿🇦", "Argentina": "🇦🇷", "Chile": "🇨🇱", "Japan": "🇯🇵",
  "South Korea": "🇰🇷", "Czech Republic": "🇨🇿", "Czechia": "🇨🇿", "Poland": "🇵🇱", "Portugal": "🇵🇹",
  "Switzerland": "🇨🇭", "Philippines": "🇵🇭", "Colombia": "🇨🇴", "Mexico": "🇲🇽", "India": "🇮🇳",
  "China": "🇨🇳", "Zimbabwe": "🇿🇼", "Thailand": "🇹🇭", "Chinese Taipei": "🇹🇼", "Taiwan": "🇹🇼",
};
const flagOf = (p) => FLAGS[p.country] || "⛳";

function byName(side, name) {
  return (DATA.players[side] || []).find((p) => p.name === name);
}

/* short display names with collision handling */
function buildShortNames() {
  DATA._short = {};
  for (const side of ["USA", "EUR"]) {
    const list = DATA.players[side] || [];
    const counts = {};
    for (const p of list) {
      const last = p.name.split(" ").slice(-1)[0];
      counts[last] = (counts[last] || 0) + 1;
    }
    for (const p of list) {
      const parts = p.name.split(" ");
      const last = parts[parts.length - 1];
      DATA._short[p.name] = counts[last] > 1 ? parts[0][0] + ". " + last : last;
    }
  }
}
const shortOf = (name) => (DATA._short && DATA._short[name]) || name;

function headHtml(p, size) {
  size = size || 26;
  if (p.headshot) {
    const init = p.name.split(" ").map((w) => w[0]).slice(0, 2).join("");
    return `<img src="${p.headshot}" width="${size}" height="${size}" alt=""
             onerror="this.outerHTML='<div class=\\'fallback\\' style=\\'width:${size}px;height:${size}px;font-size:${Math.max(9, size / 2.4)}px\\'>${init}</div>'">`;
  }
  const init = p.name.split(" ").map((w) => w[0]).slice(0, 2).join("");
  return `<div class="fallback" style="width:${size}px;height:${size}px;font-size:${Math.max(9, size / 2.4)}px">${init}</div>`;
}

function formArrow(p) {
  const f = -p.momentum; // strokes gained vs field recently
  if (f > 0.15) return `<span class="arrow-hot">▲ ${f.toFixed(1)}</span>`;
  if (f < -0.15) return `<span class="arrow-cold">▼ ${Math.abs(f).toFixed(1)}</span>`;
  return `<span class="arrow-flat">—</span>`;
}

/* ===================== overview ===================== */
let marginChart = null;

function renderOverview() {
  const s = DATA.session;
  $("p-usa").textContent = pct(s.p_usa_win);
  $("p-tie").textContent = pct(s.p_14_14);
  $("p-eu").textContent = pct(s.p_europe_win);
  $("p-usa-sub").textContent = "first side to 14½ points";
  $("p-eu-sub").textContent = "retain on a tie: " + pct(s.p_europe_retain);
  $("sim-n").textContent = "(" + s.n.toLocaleString() + " simulated cups)";

  const dist = s.us_points_dist || {};
  const keys = Object.keys(dist).map(Number).sort((a, b) => a - b);
  const labels = keys.map((k) => (k === 14 ? "14–14 tie" : "USA " + k + "–" + (28 - k)));
  const colors = keys.map((k) => (k === 14 ? "#d4af37" : k > 14 ? "#c8102e" : "#2f6fd0"));
  if (marginChart) marginChart.destroy();
  marginChart = new Chart($("marginChart"), {
    type: "bar",
    data: { labels, datasets: [{ data: keys.map((k) => dist[k]), backgroundColor: colors, borderRadius: 3 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: (c) => "Final score: " + c[0].label, label: (c) => pct(c.raw) } },
      },
      scales: {
        x: { ticks: { color: "#93a0b8", maxRotation: 90, minRotation: 45, font: { size: 10 } }, grid: { display: false } },
        y: { ticks: { color: "#93a0b8", callback: (v) => pct(v, 0) }, grid: { color: "rgba(255,255,255,.06)" } },
      },
    },
  });

  const best = keys.reduce((a, b) => (dist[a] >= dist[b] ? a : b));
  $("likely-score").innerHTML =
    "Most likely final score: <span style='color:" + (best === 14 ? "#d4af37" : best > 14 ? "#ff6b83" : "#7fb0ff") + "'>" +
    (best === 14 ? "14 – 14 (Europe retains)" : "USA " + best + " – " + (28 - best) + " Europe") +
    "</span> at " + pct(dist[best]);

  const q = DATA.meta.qualification;
  $("us-frac").textContent = Math.round(q.usa.fraction_complete * 100) + "% complete";
  $("us-bar").style.width = Math.max(1.5, q.usa.fraction_complete * 100) + "%";
  $("us-rules").textContent = q.usa.rules;
  $("eu-frac").textContent = Math.round(q.eur.fraction_complete * 100) + "% complete";
  $("eu-bar").style.width = Math.max(1.5, q.eur.fraction_complete * 100) + "%";
  $("eu-rules").textContent = q.eur.rules;

  const m = DATA.meta;
  const bits = [
    "Model generated <b>" + m.generated_at.slice(0, 16).replace("T", " ") + " UTC</b>",
    "fed by <b>" + m.data.events_counted + "</b> completed tour events",
    m.data.manual_events_counted + " hand-logged DP World Tour results",
    "OWGR as of " + m.owgr_as_of,
    m.eu_points_as_of ? "official Europe points snapshot " + m.eu_points_as_of : null,
  ].filter(Boolean);
  let note = "<div class='panel-title'>Data freshness</div><p class='small'>" + bits.join(" · ") + ".</p>";
  if (m.warnings && m.warnings.length) {
    note += "<p class='small' style='color:#e05252'>⚠ " + m.warnings.join(" · ") + "</p>";
  }
  if (m.discovered && m.discovered.length) {
    note += "<p class='muted small'>Auto-discovered candidates: " + m.discovered.join(", ") + "</p>";
  }
  note += "<p class='muted small'>The GitHub Action refreshes this data twice a week. As 2027 tournaments finish, ratings, teams, matchups and scorelines shift with them.</p>";
  $("data-note").innerHTML = note;
}

/* ===================== teams ===================== */
function chipHtml(p) {
  return `<div class="chip" title="${p.name} — OWGR #${p.owgr_rank ?? "—"}, rating ${p.rating}">
    ${headHtml(p, 26)}<span class="flag">${flagOf(p)}</span><span class="nm">${shortOf(p.name)}</span></div>`;
}

function renderTeams() {
  for (const [side, id, cls] of [["USA", "team-usa", "us"], ["EUR", "team-eur", "eu"]]) {
    const t = DATA.teams[side];
    const auto = t.projected_auto.map((n) => byName(side, n)).filter(Boolean);
    const picks = t.projected_picks.map((n) => byName(side, n)).filter(Boolean);
    $(id).innerHTML = `
      <div class="team-head">TEAM ${side === "USA" ? "USA" : "EUROPE"}
        <div class="team-captain">captain: ${t.captain}</div></div>
      <div class="team-body">
        <div class="slot-label">Projected automatic qualifiers (points race)</div>
        <div class="chip-grid">${auto.map(chipHtml).join("")}</div>
        <div class="slot-label">Projected captain's picks</div>
        <div class="chip-grid">${picks.map(chipHtml).join("")}</div>
      </div>`;
  }
  renderBubbles("USA", "bubble-usa");
  renderBubbles("EUR", "bubble-eur");
}

function renderBubbles(side, id) {
  const list = DATA.players[side] || [];
  const inSquad = new Set([...DATA.teams[side].projected_auto, ...DATA.teams[side].projected_picks]);
  const rows = list.map((p) => {
    const pts = p.points > 0 ? Math.round(p.points).toLocaleString() : "—";
    const src = p.points_source === "official" ? `<span class="tag official">official</span>`
      : p.points_source === "modeled" ? `<span class="tag">modeled</span>` : "";
    return `<tr class="${inSquad.has(p.name) ? "in" : ""}">
      <td>${headHtml(p, 22)}</td>
      <td><b>${p.name}</b> ${flagOf(p)}
        ${inSquad.has(p.name) ? '<span class="badge in">IN</span>' : ""}
        ${p.liv ? '<span class="badge liv">LIV</span>' : ""}</td>
      <td>${p.owgr_rank ?? "—"}</td>
      <td>${p.rating.toFixed(2)}</td>
      <td>${formArrow(p)}</td>
      <td>${pts} ${src}</td>
      <td>${pct(p.p_auto, 0)}</td>
      <td>${pct(p.p_pick, 0)}</td>
      <td><div class="pteam-bar"><i style="width:${(p.p_team * 100).toFixed(1)}%;
        background:${side === "USA" ? "linear-gradient(90deg,#c8102e,#ff6b83)" : "linear-gradient(90deg,#2f6fd0,#7fb0ff)"}"></i></div></td>
      <td>${pct(p.p_team, 0)}</td>
      <td>${p.apps || 0}</td>
    </tr>`;
  }).join("");
  $(id).innerHTML = `<div style="overflow-x:auto"><table class="bubble">
    <thead><tr><th></th><th>Player</th><th>OWGR</th><th>Rating</th><th>Form</th><th>Points</th>
    <th>P(auto)</th><th>P(pick)</th><th colspan="2">P(team)</th><th>RC</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

/* ===================== match lab ===================== */
let scorelineChart = null;
let matchFmt = "singles";
const sel = { USA: [0, 1], EUR: [0, 1] };

function playerOptions(side, idx) {
  const list = DATA.players[side];
  return list.map((p, i) =>
    `<option value="${i}" ${i === sel[side][idx] ? "selected" : ""}>${p.name} (#${p.owgr_rank ?? "—"}, ${p.rating.toFixed(1)})</option>`
  ).join("");
}

function renderMatchSelects() {
  const pair = matchFmt === "singles" ? [0] : [0, 1];
  const grp = (side, idx) => `
    <div class="sel-group ${side === "USA" ? "us" : "eu"}">
      <span class="sel-label">${side === "USA" ? "USA" : "Europe"} ${pair.length > 1 ? "player " + (idx + 1) : "player"}</span>
      <select data-side="${side}" data-idx="${idx}">${playerOptions(side, idx)}</select>
    </div>`;
  let html = "";
  for (const idx of pair) { html += grp("USA", idx); }
  for (const idx of pair) { html += grp("EUR", idx); }
  $("match-selects").innerHTML = html;
  document.querySelectorAll("#match-selects select").forEach((s) => {
    s.addEventListener("change", () => {
      sel[s.dataset.side][+s.dataset.idx] = +s.value;
      runMatchLab();
    });
  });
}

function currentDists() {
  const usList = DATA.players.USA, euList = DATA.players.EUR;
  const us1 = usList[sel.USA[0]], us2 = usList[sel.USA[1]];
  const eu1 = euList[sel.EUR[0]], eu2 = euList[sel.EUR[1]];
  if (matchFmt === "singles") {
    return { mode: "singles", a: distOf(us1), b: distOf(eu1), us1, eu1 };
  }
  if (matchFmt === "fourball") {
    return { mode: "fourball", a1: distOf(us1), a2: distOf(us2), b1: distOf(eu1), b2: distOf(eu2), us1, us2, eu1, eu2 };
  }
  return { mode: "foursomes", a: foursomesDistOf(us1, us2), b: foursomesDistOf(eu1, eu2), us1, us2, eu1, eu2 };
}

function runMatchLab() {
  const d = currentDists();
  const res = d.mode === "fourball"
    ? simFourball(d.a1, d.a2, d.b1, d.b2, MATCH_N)
    : simSingles(d.a, d.b, MATCH_N);
  const usName = d.mode === "singles" ? d.us1.name : d.us1.name + " & " + d.us2.name;
  const euName = d.mode === "singles" ? d.eu1.name : d.eu1.name + " & " + d.eu2.name;
  renderMatchResult($("match-result"), res, usName, euName, true);
  renderScorelineChart(res, usName, euName);
}

function renderMatchResult(container, res, usName, euName, withTiles) {
  const n = res.n;
  const pUs = res.nWin / n, pEu = res.nLose / n, pX = res.nExtra / n;
  const winEntries = Object.entries(res.win).sort((a, b) => b[1] - a[1]);
  const loseEntries = Object.entries(res.lose).sort((a, b) => b[1] - a[1]);
  const topWin = winEntries[0], topLose = loseEntries[0];
  let mostLikely = "—";
  if (topWin && (!topLose || topWin[1] >= topLose[1])) mostLikely = `USA by ${topWin[0].replace(" (A)", "")} (${pct(topWin[1] / n)})`;
  else if (topLose) mostLikely = `Europe by ${topLose[0].replace(" (B)", "")} (${pct(topLose[1] / n)})`;
  container.innerHTML = withTiles ? `
    <div class="match-result">
      <div class="match-tile us"><div class="who">${usName}</div><div class="pct">${pct(pUs)}</div>
        <div class="sub">wins the match${topWin ? " · most likely " + topWin[0].replace(" (A)", "") : ""}</div></div>
      <div class="match-tile x"><div class="who">Extra holes</div><div class="pct">${pct(pX)}</div>
        <div class="sub">all square after 18</div></div>
      <div class="match-tile eu"><div class="who">${euName}</div><div class="pct">${pct(pEu)}</div>
        <div class="sub">wins the match${topLose ? " · most likely " + topLose[0].replace(" (B)", "") : ""}</div></div>
      <div class="most-likely">Most likely exact scoreline: <b>${mostLikely}</b></div>
    </div>` : "";
}

function renderScorelineChart(res, usName, euName) {
  const n = res.n;
  const entries = [];
  for (const [k, v] of Object.entries(res.win)) entries.push({ label: "USA " + k.replace(" (A)", ""), v: v / n, side: "us" });
  for (const [k, v] of Object.entries(res.lose)) entries.push({ label: "EUR " + k.replace(" (B)", ""), v: v / n, side: "eu" });
  entries.sort((a, b) => b.v - a.v);
  const keep = entries.slice(0, 14);
  const rest = entries.slice(14);
  if (rest.length) {
    keep.push({ label: "other", v: rest.reduce((s, e) => s + e.v, 0), side: "mid" });
  }
  if (scorelineChart) scorelineChart.destroy();
  scorelineChart = new Chart($("scorelineChart"), {
    type: "bar",
    data: {
      labels: keep.map((e) => e.label),
      datasets: [{
        data: keep.map((e) => e.v),
        backgroundColor: keep.map((e) => e.side === "us" ? "#c8102e" : e.side === "eu" ? "#2f6fd0" : "#5a6678"),
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: "Exact scoreline distribution — " + usName + " vs " + euName, color: "#e8ecf4", font: { size: 13 } },
        tooltip: { callbacks: { label: (c) => pct(c.raw) } },
      },
      scales: {
        x: { ticks: { color: "#93a0b8", callback: (v) => pct(v, 0) }, grid: { color: "rgba(255,255,255,.06)" } },
        y: { ticks: { color: "#c6cfe0", font: { size: 11 } }, grid: { display: false } },
      },
    },
  });
}

/* ===================== lineups ===================== */
function heatColor(p) {
  const alpha = Math.abs(p - 0.5) * 2;
  const a = Math.max(0.10, alpha * 0.75);
  return p > 0.5 ? `rgba(200,16,46,${a.toFixed(2)})` : `rgba(47,111,208,${a.toFixed(2)})`;
}

function renderHeatmap() {
  const m = DATA.matrix;
  let html = "<table class='heat'><thead><tr><th class='row-label'>USA ↓ / Europe →</th>";
  for (const n of m.eu) html += `<th title="${n}">${shortOf(n)}</th>`;
  html += "</tr></thead><tbody>";
  m.us.forEach((un, i) => {
    html += `<tr><th class="row-label" title="${un}">${shortOf(un)}</th>`;
    m.eu.forEach((en, j) => {
      const c = m.cells[i][j];
      html += `<td class="cell" data-i="${i}" data-j="${j}" style="background:${heatColor(c.w)}"
                title="${un} vs ${en}: USA ${pct(c.w)}${c.x > 0.02 ? " · extra holes " + pct(c.x, 0) : ""}">
                ${Math.round(c.w * 100)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  $("heatmap").innerHTML = html;
  document.querySelectorAll("#heatmap td.cell").forEach((td) => {
    td.addEventListener("click", () => {
      document.querySelectorAll("#heatmap td.cell").forEach((x) => x.classList.remove("sel"));
      td.classList.add("sel");
      const i = +td.dataset.i, j = +td.dataset.j;
      const un = DATA.matrix.us[i], en = DATA.matrix.eu[j];
      const pu = byName("USA", un), pe = byName("EUR", en);
      const res = simSingles(distOf(pu), distOf(pe), 25000);
      const winEntries = Object.entries(res.win).sort((a, b) => b[1] - a[1]).slice(0, 5);
      const loseEntries = Object.entries(res.lose).sort((a, b) => b[1] - a[1]).slice(0, 5);
      const bar = (lab, v, side) => `<div class="dream-row"><span>${lab}</span>
        <span class="pct ${side === "us" ? "good" : "bad"}">${pct(v)}</span></div>`;
      let rows = "";
      for (const [k, v] of winEntries) rows += bar("USA " + k.replace(" (A)", ""), v / res.n, "us");
      for (const [k, v] of loseEntries) rows += bar("EUR " + k.replace(" (B)", ""), v / res.n, "eu");
      $("heat-detail").innerHTML = `
        <div class="panel" style="margin-top:14px">
          <div class="panel-title">${un} <span class="muted">vs</span> ${en}
            <span class="muted small">— 25,000 fresh simulations</span></div>
          <p><b style="color:#ff6b83">USA wins ${pct(res.nWin / res.n)}</b> ·
             <b style="color:#7fb0ff">Europe wins ${pct(res.nLose / res.n)}</b> ·
             <b style="color:#d4af37">extra holes ${pct(res.nExtra / res.n)}</b></p>
          <div class="grid-2">${rows}</div>
        </div>`;
    });
  });
}

function renderBoards() {
  const mk = (board, title, id, accent) => {
    const items = board.map((r) => `<li><span><b>${r.us}</b> <span class="vs">vs</span> ${r.eu}</span>
      <span class="p" style="color:${accent}">${pct(r.p_us_win)}</span></li>`).join("");
    $(id).innerHTML = `<div class="panel-title">${title}</div>
      <ul class="board-list">${items}</ul>
      <p class="muted small" style="margin-top:8px">Expected USA points from this board:
      <b>${board.reduce((s, r) => s + r.p_us_win, 0).toFixed(2)}</b> of 12</p>`;
  };
  mk(DATA.lineups.us_board, "If USA controls the draw — its ideal board", "us-board", "#ff6b83");
  mk(DATA.lineups.eu_board, "If Europe controls the draw — its ideal board", "eu-board", "#7fb0ff");

  const dreamUs = [], dreamEu = [];
  for (const r of DATA.lineups.best_worst) {
    if (r.us) dreamUs.push(r);
    else dreamEu.push(r);
  }
  const dreamRow = (r) => {
    const isUs = !!r.us;
    const name = isUs ? r.us : r.eu;
    const goodP = isUs ? r.best.p_us_win : r.best.p_eu_win;
    const badP = isUs ? r.worst.p_us_win : r.worst.p_eu_win;
    return `<div class="dream-row">
      <span><b>${name}</b><br><span class="matchup">dream: ${r.best.vs} · nightmare: ${r.worst.vs}</span></span>
      <span class="pct good">${pct(goodP)}</span><span class="pct bad">${pct(badP)}</span></div>`;
  };
  $("dream-usa").innerHTML = dreamUs.map(dreamRow).join("");
  $("dream-eur").innerHTML = dreamEu.map(dreamRow).join("");

  const pairsHtml = (side) => {
    const p = DATA.pairings[side];
    const one = (rows, fmtName) => rows.map((x) =>
      `<div class="pair-card"><span><span class="fmt-tag">${fmtName}</span>${x.players.join(" & ")}</span>
       <span class="exp">${pct(x.expected_points)}</span></div>`).join("");
    return one(p.fourball, "four-ball") + one(p.foursomes, "foursomes");
  };
  $("pairs-usa").innerHTML = `<div class="panel-title">USA</div>` + pairsHtml("USA");
  $("pairs-eur").innerHTML = `<div class="panel-title">Europe</div>` + pairsHtml("EUR");
}

/* ===================== about ===================== */
function renderAbout() {
  const m = DATA.meta;
  $("about-data").innerHTML =
    `Right now that's <b>${m.data.events_counted}</b> completed events, OWGR as of ${m.owgr_as_of}` +
    (m.eu_points_as_of ? `, official Europe points as of ${m.eu_points_as_of}` : "") + ".";
  $("about-params").innerHTML =
    `Calibration: match sharpness ${m.model.match_sharpness}× (a 2-stroke favourite wins ~77% of singles), ` +
    `home advantage ${m.model.home_advantage_strokes} strokes/round for Europe in the cup simulation, ` +
    `${m.model.form_half_life_weeks}-week form half-life, ${m.model.selection_sim_sims.toLocaleString()} selection simulations, ` +
    `${DATA.session.n.toLocaleString()} full-cup simulations.`;
  const repo = $("repo-link");
  if (REPO_URL) { repo.href = REPO_URL; repo.textContent = "★ View the source & model on GitHub"; }
  else repo.style.display = "none";
}

/* ===================== chrome ===================== */
function renderChrome() {
  const m = DATA.meta;
  const start = new Date(m.event.start + "T06:00:00Z");
  const tick = () => {
    const ms = start - Date.now();
    const days = Math.ceil(ms / 86400000);
    $("countdown").textContent = days > 1 ? days + " days to the first tee"
      : days === 1 ? "1 day to the first tee" : days === 0 ? "It's Ryder Cup week!" : "The 46th Ryder Cup is underway";
  };
  tick();
  setInterval(tick, 60000);
  $("updated").textContent = "model updated " + m.today;
  $("venue-line").textContent = `${m.event.venue} · 17–19 September 2027 · a self-updating prediction model`;
  $("foot-generated").textContent = "model generated " + m.generated_at.slice(0, 10);
}

function setupTabs() {
  document.querySelectorAll("#tabs button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#tabs button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      $("tab-" + btn.dataset.tab).classList.add("active");
      window.scrollTo({ top: 0 });
    });
  });
}

/* ===================== boot ===================== */
function init() {
  setupTabs();
  document.querySelectorAll("#fmt-toggle button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#fmt-toggle button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      matchFmt = b.dataset.fmt;
      renderMatchSelects();
      runMatchLab();
    });
  });
  $("reroll").addEventListener("click", runMatchLab);
  fetch("data/model_output.json")
    .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then((d) => {
      DATA = d;
      SHARP = d.meta.model.match_sharpness || 1.9;
      buildShortNames();
      renderChrome();
      renderOverview();
      renderTeams();
      renderMatchSelects();
      runMatchLab();
      renderHeatmap();
      renderBoards();
      renderAbout();
    })
    .catch((err) => {
      document.querySelector("main").innerHTML =
        `<div class="panel"><div class="panel-title">Could not load model output</div>
         <p class="muted">${err}</p>
         <p class="muted small">Run <code>python3 -m model.pipeline</code> first, and serve this
         folder over HTTP (e.g. <code>python3 -m http.server</code>) rather than opening the file directly.</p></div>`;
    });
}

document.addEventListener("DOMContentLoaded", init);
