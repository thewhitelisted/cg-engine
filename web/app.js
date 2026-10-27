/* The Cool Guys league page, built in the browser from the JSON API.

   The page loads api/state.json (everything the page shows), draws it, then listens on api/events: the server pushes
   a small patch after every change and the page redraws only what moved (morph: the elements on screen are kept, so
   scroll position and open charts survive). If the stream drops, the browser reconnects; a full refresh of the state
   runs whenever the page falls behind or nothing has arrived for a minute.

   render(state) is a line-by-line port of the page's original Python renderer (league_page.render, retired
   2026-10-25); tests/test_page_render.py holds the page it produced on fixed states, and this must match it exactly.
   In Node (tests), module.exports gives {render, applyPatch, morph, fixed, pct}. */
(function (root) {
  "use strict";
  const CATS = ["FG%", "FT%", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"];
  const NAV = [["matchups", "Matchups"], ["today", "Today"], ["awards", "Awards"], ["standings", "Standings"],
    ["weeks", "Weekly"], ["ranks", "Cat ranks"], ["h2h", "Head to head"], ["allplay", "All-play"], ["records", "Records"],
    ["top", "Best players"], ["worst", "Worst players"], ["odds", "Odds"], ["moves", "Moves"], ["activity", "Activity"]];

  // ------------------------------------------------------------ Python-compatible formatting
  const e = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#x27;");                   // html.escape
  function fixedAbs(a, d) {                 // f"{a:.{d}f}" for a >= 0: nearest, exact halves to even
    const s = a.toFixed(Math.min(100, d + 40));                           // the double's exact decimal digits
    const cut = s.indexOf(".") + 1 + d;
    if (!/^50*$/.test(s.slice(cut))) return a.toFixed(d);
    const digits = (d ? s.slice(0, cut) : s.slice(0, cut - 1)).replace(".", "");
    const n = (BigInt(digits) + (BigInt(digits) % 2n)).toString().padStart(d + 1, "0");
    return d ? n.slice(0, -d) + "." + n.slice(-d) : n;
  }
  const neg = x => x < 0 || Object.is(x, -0);
  const fixed = (x, d) => (neg(x) ? "-" : "") + fixedAbs(Math.abs(x), d);              // f"{x:.df}"
  const signed = (x, d) => (neg(x) ? "-" : "+") + fixedAbs(Math.abs(x), d);           // f"{x:+.df}"
  const pct = (x, d = 0) => fixed(x * 100, d) + "%";                                  // f"{x:.d%}"
  const g6 = x => String(Number(x.toPrecision(6)));                                    // f"{x:g}"
  const pyNum = x => (Number.isInteger(x) ? x.toFixed(1) : String(x));               // json.dumps float
  const pyJson = pts => "[" + pts.map(([t, p]) => `[${JSON.stringify(t)}, ${pyNum(p)}]`).join(", ") + "]";

  // ------------------------------------------------------------ page sections
  const section = (title, body, note = "", sid = "") =>
    `<section id="${sid}" style="display:grid;gap:14px"><h2>${title}</h2><div class="card tbl scroll">${body}</div>` +
    (note ? `<p class="note">${note}</p>` : "") + "</section>";

  function trendBlock(trend, start, end, ha, aa, key = "") {
    if (!trend.length) return "";
    const attrs = `data-trend="${e(pyJson(trend))}" data-start="${start}" data-end="${end}" data-ha="${ha}" data-aa="${aa}"`;
    return `<details class="trend" data-key="${e(key)}"><summary aria-label="Odds history chart">` +
      `<div class="mini" ${attrs} aria-hidden="true"></div>` +
      `<span class="pill"><span class="closed">Expand chart</span><span class="open">Collapse</span>` +
      `<span class="chev">&#8964;</span></span></summary>` +
      `<div class="tplot" ${attrs}></div></details>`;
  }

  function card(gm, start, end) {
    const ph = gm.p_home;
    const [w, l, t] = gm.score;
    const [[hp, hl], [ap, al]] = gm.games;
    const ha = e(gm.home_abbr || gm.home.slice(0, 3).toUpperCase()), aa = e(gm.away_abbr || gm.away.slice(0, 3).toUpperCase());
    const head = CATS.map(c => `<th>${c}</th>`).join("");
    const rh = gm.totals.map(([, h], i) => `<td class="${gm.lead[i] === 1 ? "lead ha" : ""}">${h}</td>`).join("");
    const ra = gm.totals.map(([, , a], i) => `<td class="${gm.lead[i] === -1 ? "lead hb" : ""}">${a}</td>`).join("");
    const bits = ["chance to win the week"];
    const sw = gm.swing;
    if (sw != null && Math.abs(sw) >= 0.01)
      bits.push(`<span class="${sw > 0 ? "ha" : "hb"}">${sw > 0 ? ha : aa} +${fixed(100 * Math.abs(sw), 0)}</span> since yesterday`);
    if (t) bits.push(`${t} tied`);
    const ip = gm.in_play;
    let inplay = "";
    if (ip && ip.length)
      inplay = `<div class="inplay"><span class="lbl">Toss-ups</span>` +
        ip.map(([c, txt]) => `<span class="chip"><b>${e(c)}</b> ${e(txt)}</span>`).join("") + "</div>";
    else if (ip != null) inplay = '<div class="inplay"><span class="lbl">No toss-up categories left.</span></div>';
    return `<div class="card" data-key="${e(gm.key || "")}">` +
      `<div class="matchup">` +
      `<div class="side"><div class="team ha">${e(gm.home)}</div><div class="pct ha">${pct(ph)}</div>` +
      `<div class="meta">leads ${w} of 9 · ${hp} played, ${hl} left</div></div>` +
      `<div class="side r"><div class="team hb">${e(gm.away)}</div><div class="pct hb">${pct(1 - ph)}</div>` +
      `<div class="meta">leads ${l} of 9 · ${ap} played, ${al} left</div></div>` +
      `</div>` +
      `<div class="bar" title="${ha} ${pct(ph, 1)} · ${aa} ${pct(1 - ph, 1)}"><i style="width:${fixed(100 * ph, 1)}%"></i>` +
      `<i style="width:${fixed(100 * (1 - ph), 1)}%"></i></div>` +
      `<div class="caption">${bits.join(" · ")}</div>` +
      trendBlock(gm.trend || [], start, end, ha, aa, gm.key || "") +
      `<div class="scroll"><table class="cats"><tr><th></th>${head}</tr>` +
      `<tr><td><span class="dot da"></span>${ha}</td>${rh}</tr>` +
      `<tr><td><span class="dot db"></span>${aa}</td>${ra}</tr></table></div>` +
      inplay +
      `</div>`;
  }

  const wlt = (w, l, t) => `${w}–${l}${t ? `–${t}` : ""}`;

  function render(data) {
    const finished = data.standings.some(r => r.w + r.l + r.t);

    let headline = "";
    const moved = data.games.filter(g => g.swing != null && Math.abs(g.swing) >= 0.05);
    if (moved.length) {
      const g = moved.reduce((a, b) => (Math.abs(b.swing) > Math.abs(a.swing) ? b : a));
      const sw = g.swing, p = g.p_home;
      const [team, now, before] = sw > 0 ? [g.home, p, p - sw] : [g.away, 1 - p, 1 - p + sw];
      headline = `<div class="headline">Biggest swing since yesterday: <b>${e(team)}</b> ${pct(before)} → ${pct(now)}</div>`;
    }

    let today = "";
    if (data.today && data.today.some(r => r.n)) {
      const rows = data.today.map(r => {
        const who = r.players.map(p => `<span class="${p.out ? "out" : p.bench ? "bn" : ""}">${e(p.name)} ` +
          `${e(p.when)}${p.bench && !p.out ? " (bench)" : ""}</span>`).join("");
        return `<tr><td>${e(r.team)}</td><td>${r.lineup}</td><td>${r.n}</td><td class="who">${who || "—"}</td></tr>`;
      });
      today = section("Playing today", `<table><tr><th>Team</th><th>In lineup</th><th>With a game</th><th class="l">Who · tip-off</th></tr>` +
        rows.join("") + "</table>",
        "In lineup: players with a game today in an active slot and not ruled out. Bench in grey, " +
        "ruled out struck through. Times Eastern.", "today");
    } else if (data.today != null) {
      today = '<section id="today"><h2>Playing today</h2><p class="note">No NBA games today.</p></section>';
    }

    let standings = "";
    if (finished) {
      const st = data.standings.map((r, i) => `<tr><td>${i + 1}. ${e(r.team)}</td><td>${r.w}–${r.l}–${r.t}</td><td>${fixed(r.pct, 3)}</td>` +
        `<td>${r.gb === 0 ? "—" : g6(r.gb)}</td></tr>`).join("");
      standings = section("Standings", `<table><tr><th>Team</th><th>Categories W–L–T</th><th>Pct</th><th>GB</th></tr>${st}</table>`, "", "standings");
    }

    let grid = "";
    const gd = data.grid;
    if (gd) {
      const head = gd.weeks.map(w => `<th>${w}</th>`).join("");
      const rows = gd.rows.map(r => {
        const cells = r.cells.map(c => (c == null ? "<td></td>" :
          `<td class="${c[4] ? "g" + c[3] : "live"}">${c[0]}–${c[1]}${c[2] ? `–${c[2]}` : ""}</td>`)).join("");
        return `<tr><td>${e(r.team)}</td><td>${r.rec[0]}–${r.rec[1]}${r.rec[2] ? `–${r.rec[2]}` : ""}</td>${cells}</tr>`;
      });
      grid = section("Week by week", `<table class="grid"><tr><th>Team</th><th>Matchups</th>${head}</tr>${rows.join("")}</table>`,
        "Category score in each matchup. Green won, red lost; grey italic is this week, still in progress.", "weeks");
    }

    let h2h = "";
    const hh = data.h2h;
    if (hh) {
      const head = hh.cols.map(c => `<th>${e(c)}</th>`).join("");
      const rows = hh.rows.map(r => {
        const cells = r.cells.map(c => {
          if (c == null) return "<td></td>";
          const [w, l, t] = c;
          return `<td class="${w > l ? "gw" : l > w ? "gl" : "gt"}">${wlt(w, l, t)}</td>`;
        }).join("");
        return `<tr><td>${e(r.team)} <span class="lbl">${e(r.abbr)}</span></td>${cells}</tr>`;
      });
      h2h = section("Head to head", `<table class="grid"><tr><th>Team</th>${head}</tr>${rows.join("")}</table>`,
        "Each row team's matchup record against each opponent.", "h2h");
    }

    let allplay = "";
    if (data.allplay && data.allplay.length) {
      const ap = data.allplay.map(r => `<tr><td>${e(r.team)}</td><td>${r.mw}–${r.ml}–${r.mt}</td><td>${fixed(r.allplay_pct, 3)}</td>` +
        `<td>${fixed(r.actual_pct, 3)}</td><td class="${r.luck > 0 ? "pos" : r.luck < 0 ? "neg" : ""}">` +
        `${signed(100 * r.luck, 1)}</td></tr>`).join("");
      allplay = section("All-play record and luck",
        `<table><tr><th>Team</th><th>All-play W–L–T</th><th>All-play cat pct</th><th>Actual cat pct</th>` +
        `<th>Luck</th></tr>${ap}</table>`,
        "All-play: every team against every other team, every finished week. Luck: actual category " +
        "win percentage minus all-play, in points. Positive means your schedule has been kind.", "allplay");
    }

    let records = "";
    if (data.records && data.records.length) {
      const rr = data.records.map(r => `<tr><td>${e(r.what)}</td><td class="l">${e(r.team)}</td><td>${e(r.value)}</td><td>${r.week}</td></tr>`).join("");
      records = section("League records", `<table><tr><th>Record</th><th class="l">Team</th><th>Value</th><th>Matchup</th></tr>${rr}</table>`,
        "Best single matchups so far. Percentages need at least 75% of the league's average attempts; " +
        "counting stats leave out the two-week All-Star matchup.", "records");
    }

    const playersTable = (rows, title, sid, note) => {
      if (!rows || !rows.length) return "";
      const tr = rows.map(r => `<tr><td>${e(r.player)}</td><td class="l">${e(r.team)}</td><td>${r.g}</td>` +
        `<td class="${(r.pct || 0) > 0 ? "pos" : "neg"}">${r.pct == null ? "—" : `${signed(100 * r.pct, 0)}%`}</td>` +
        `<td class="l">${e(r.line)}</td></tr>`).join("");
      return section(title, `<table><tr><th>Player</th><th class="l">Fantasy team</th><th>G</th><th>vs expected</th>` +
        `<th class="l">Line</th></tr>${tr}</table>`, note, sid);
    };
    const how = "vs expected: 9-category production against what the model expected from the same number of games. " +
      "Ranked by the size of the gap, so a big game from a big role outranks a big percentage from a small one. " +
      "Rostered players, games through last night.";
    const top = playersTable(data.top, "Best performers this week", "top", how);
    const worst = playersTable(data.worst || [], "Worst performers this week", "worst",
      how + " Below −100%: turnovers and missed shots outweighed everything else.");

    let last = "";
    if (data.last) {
      const lr = data.last.rows.map(r => {
        const p = r.p_home;
        const odds = p == null ? "—" : `${pct(p)} / ${pct(1 - p)}`;
        const win = r.tie ? "Tie" : r.home_won ? r.home : r.away;
        const favWon = p != null && !r.tie && ((p >= 0.5) === r.home_won);
        return `<tr><td>${e(r.home)} vs ${e(r.away)}</td><td>${odds}</td><td>${r.score[0]}–${r.score[1]}–${r.score[2]}</td>` +
          `<td class="l">${e(win)}${p == null || r.tie ? "" : favWon ? " ✓" : " (upset)"}</td></tr>`;
      });
      last = section(`Matchup ${data.last.period}: odds against results`,
        `<table><tr><th>Matchup</th><th>Odds at first update</th><th>Final</th><th class="l">Winner</th></tr>` +
        lr.join("") + "</table>", "", "odds");
    }

    let score = "";
    const sc = data.scorecard;
    if (sc) {
      const tiles = [["Favorites", `${sc.fav[0]}–${sc.fav[1]}`, "won–lost, ties left out"],
        ["Strong favorites", `${sc.big[0]}–${sc.big[1]}`, "75% or more"],
        ["Brier score", fixed(sc.brier, 3), "coin flip .250 · lower is better"],
        ["Avg. confidence", pct(sc.conf), `in the favorite, ${sc.n} matchups`]];
      score = `<section id="${last ? "" : "odds"}" style="display:grid;gap:14px"><h2>How the odds have done</h2><div class="tiles">` +
        tiles.map(([a, b, c]) => `<div class="card tile"><div class="lbl">${a}</div><div class="big">${b}</div><div class="meta">${c}</div></div>`).join("") +
        '</div><p class="note">Every finished matchup, scored on the odds at its first update of the week.</p></section>';
    }

    let moves = "";
    if (data.moves && data.moves.length) {
      const mv = data.moves.map(r => (r.kind === "trade"
        ? `<tr><td>${e(r.when)}</td><td class="l">Trade</td><td class="who" colspan="2">${e(r.add)}</td></tr>`
        : `<tr><td>${e(r.when)}</td><td class="l">${e(r.team)} <span class="lbl">· ${e(r.kind)}</span></td>` +
          `<td class="l">${e(r.add) || "—"}</td><td class="l">${e(r.drop) || "—"}</td></tr>`));
      moves = section("Recent moves", `<table><tr><th>When</th><th class="l">Team</th><th class="l">Added</th>` +
        `<th class="l">Dropped</th></tr>${mv.join("")}</table>`, "Executed adds, drops and trades from the last 7 days.", "moves");
    }

    let awards = "";
    const aw = data.awards;
    if (aw)
      awards = `<section id="awards" style="display:grid;gap:14px"><h2>Matchup ${aw.period} awards</h2><div class="tiles">` +
        aw.items.map(a => `<div class="card tile award"><div class="lbl">${e(a.title)}</div><div class="big">${e(a.team)}</div>` +
          `<div class="meta">${e(a.text)}</div></div>`).join("") + "</div></section>";

    let ranks = "";
    const rk = data.ranks;
    if (rk) {
      const n = rk.n;
      const heat = r => {
        const x = n > 1 ? (n - r) / (n - 1) : 0.5;
        const [col, amt] = x >= 0.5 ? ["--good", (x - 0.5) * 2] : ["--bad", (0.5 - x) * 2];
        return `background: color-mix(in srgb, var(${col}) ${fixed(6 + 34 * amt, 0)}%, transparent)`;
      };
      const head = CATS.map(c => `<th>${c}</th>`).join("");
      const rows = rk.rows.map(r => `<tr><td>${e(r.team)}</td>` +
        r.ranks.map((k, i) => `<td class="rk" style="${heat(k)}" title="${e(r.shown[i])}">${k}</td>`).join("") +
        `<td>${fixed(r.avg, 1)}</td></tr>`).join("");
      ranks = section("Category ranks", `<table class="grid"><tr><th>Team</th>${head}<th>Avg</th></tr>${rows}</table>`,
        "Season rank in each category over finished matchups (1 is best; fewest turnovers ranks first). " +
        "Hover or long-press a cell for the season figure.", "ranks");
    }

    let activity = "";
    if (data.activity && data.activity.length) {
      const ac = data.activity.map(r => `<tr><td>${e(r.team)}</td><td>${r.adds}</td><td>${r.drops}</td><td>${r.trades}</td></tr>`).join("");
      activity = section("Most active managers", `<table><tr><th>Team</th><th>Adds</th><th>Drops</th><th>Trades</th>` +
        `</tr>${ac}</table>`, "Since the draft, from ESPN's transaction log.", "activity");
    }

    const later = finished ? "" : '<p class="note">After the first matchup is final, this page adds standings, week-by-week ' +
      "results, head-to-head records, all-play and luck, league records, and a running scorecard for the odds.</p>";
    const sections = [today, awards, standings, grid, ranks, h2h, allplay, records, top, worst, last, score, moves, activity];
    const present = new Set(["matchups", ...["today", "awards", "standings", "weeks", "ranks", "h2h", "allplay", "records", "top",
      "worst", "odds", "moves", "activity"].filter(k => sections.join("").includes(`id="${k}"`))]);
    const nav = '<nav class="jump" aria-label="Sections">' +
      NAV.filter(([k]) => present.has(k)).map(([k, lbl]) => `<a href="#${k}">${lbl}</a>`).join("") + "</nav>";
    const cards = data.games.map(gm => card(gm, data.start, data.end)).join("\n  ");
    return `
  <div><h1>The Cool Guys · Matchup ${data.period}</h1>
  <div class="sub">${e(data.dates)} · updated ${e(data.updated)}</div>${headline}</div>
  ${nav}
  <div id="matchups" style="display:grid;gap:14px">${cards}</div>
  <p class="note">Win chance: each team's chance to win at least 5 of the 9 categories by Sunday, from what's happened so
  far plus a projection of every remaining game. The strip under each bar is that chance through the week (blue share
  at the bottom, one column per day). Toss-ups: categories where either side has at least a 30% chance to win.
  Games: player-games already played this week · scheduled games left. Updates live: new numbers appear on their own, without reloading.</p>
  ${today}
  ${awards}
  ${standings}
  ${grid}
  ${ranks}
  ${h2h}
  ${allplay}
  ${records}
  ${top}
  ${worst}
  ${last}
  ${score}
  ${moves}
  ${activity}
  ${later}
  <footer class="foot"><a class="btn" href="insights.html" target="_blank" rel="noopener">Insights: how the odds work ↗</a></footer>
`;
  }

  // ------------------------------------------------------------ patches (api.py writes them; same ops, same order)
  const isObj = x => x !== null && typeof x === "object" && !Array.isArray(x);
  function locate(node, step) {
    if (typeof step === "string" && step[0] === "@" && Array.isArray(node)) return node.findIndex(x => x && String(x.key) === step.slice(1));
    return step;
  }
  function applyPatch(state, ops) {         // returns a new state; ops: [op, path, value]
    let out = JSON.parse(JSON.stringify(state));
    for (const [op, path, value] of ops) {
      if (!path.length) { out = JSON.parse(JSON.stringify(value)); continue; }               // ["set", [], whole]
      let node = out;
      for (const step of path.slice(0, -1)) node = node[locate(node, step)];
      const k = locate(node, path[path.length - 1]);
      if (op === "set") node[k] = value;
      else if (op === "del") Array.isArray(node) ? node.splice(k, 1) : delete node[k];
      else if (op === "push") node[k].push(...value);
      else if (op === "order") { const arr = node[k], byKey = new Map(arr.map(x => [x.key, x])); node[k] = value.map(x => byKey.get(x)); }
    }
    return out;
  }

  // ------------------------------------------------------------ morph: update the page in place
  const key = n => (n.nodeType === 1 ? n.getAttribute("data-key") || n.id || "" : "");
  const charts = n => (n.nodeType === 1 ? [...(n.matches(".tplot, .mini") ? [n] : []), ...n.querySelectorAll(".tplot, .mini")] : []);
  const hrefs = n => [...n.querySelectorAll("a")].map(a => a.getAttribute("href")).join();
  function morph(a, b, st) {                // make a match b, keeping a's elements (and their state) wherever they line up
    const doc = a.ownerDocument;
    if (a.nodeType !== 1) { if (a.nodeValue !== b.nodeValue) a.nodeValue = b.nodeValue; return; }
    if (a.matches("nav.jump")) {            // keep the highlighted section unless the sections themselves changed
      if (hrefs(a) !== hrefs(b)) { a.replaceWith(doc.importNode(b, true)); st.nav = true; }
      return;
    }
    let changed = false;
    for (const { name } of [...a.attributes])
      if (!b.hasAttribute(name) && !(name === "open" && a.tagName === "DETAILS")) { a.removeAttribute(name); changed = true; }
    for (const { name, value } of [...b.attributes])
      if (a.getAttribute(name) !== value) { a.setAttribute(name, value); changed = true; }
    if (a.matches(".tplot, .mini")) { if (changed) st.dirty.add(a); return; }   // drawn by script from their data-*
    const keyed = new Map(), free = [];
    for (const n of a.childNodes) key(n) ? keyed.set(key(n), n) : free.push(n);
    let j = 0;
    [...b.childNodes].forEach((y, i) => {
      let x = key(y) ? keyed.get(key(y)) : free[j++];
      if (key(y)) keyed.delete(key(y));
      if (x && x.nodeName === y.nodeName) morph(x, y, st);
      else { x = doc.importNode(y, true); charts(x).forEach(c => st.dirty.add(c)); }
      if (a.childNodes[i] !== x) a.insertBefore(x, a.childNodes[i] || null);
    });
    while (a.childNodes.length > b.childNodes.length) a.removeChild(a.lastChild);
  }

  const api = { render, applyPatch, morph, fixed, pct, signed, g6 };
  if (typeof module !== "undefined" && module.exports) { module.exports = api; return; }

  // ------------------------------------------------------------ in the browser: load, draw, follow the stream
  const doc = root.document;
  const API = root.CG_API || "";             // the live server (set by the page); this site's own copy is the fallback
  let state = null, first = true, lastHeard = Date.now();
  function paint() {
    const t = doc.createElement("template");
    t.innerHTML = `<div class="wrap">${render(state)}</div>`;
    const st = { dirty: new Set(), nav: false };
    morph(doc.querySelector(".wrap"), t.content.firstElementChild, st);
    CG.bindCharts();                          // CG and initNav: the page's own script (shared global scope)
    if (first) { first = false; initNav(); CG.initCharts(); return; }
    st.dirty.forEach(box => (box.matches(".tplot") ? box.closest("details.trend")?.open && CG.draw(box) : CG.mini(box)));
    if (st.nav) initNav();
  }
  function show(s) {                        // a whole state: take it unless it's older than what's on screen
    lastHeard = Date.now();
    if (state && (s.ts || 0) < state.ts) return;
    if (state && s.v === state.v) return;
    state = s;
    paint();
  }
  async function load() {
    for (const base of API ? [API, ""] : [""]) {
      try {
        const r = await fetch(base + "api/state.json?t=" + Date.now(), { cache: "no-store" });
        if (!r.ok) throw new Error(r.status);
        return show(await r.json());
      } catch (err) { /* try the next source; the poll retries */ }
    }
  }
  function follow() {
    if (!("EventSource" in root)) return;
    const es = new root.EventSource(API + "api/events?v=" + encodeURIComponent(state ? state.v : ""));
    es.addEventListener("patch", ev => {
      lastHeard = Date.now();
      const m = JSON.parse(ev.data);
      if (state && m.v === state.v) return;                                  // already have it
      if (!state || m.from !== state.v) return load();                       // behind: fetch everything
      state = Object.assign(applyPatch(state, m.ops), { v: m.v, ts: m.ts });
      paint();
    });
    es.addEventListener("state", ev => show(JSON.parse(ev.data)));
    es.addEventListener("ping", () => { lastHeard = Date.now(); });
  }
  setInterval(() => { if (!doc.hidden && Date.now() - lastHeard > 60000) load(); }, 15000);   // stream quiet: poll
  doc.addEventListener("visibilitychange", () => { if (!doc.hidden && Date.now() - lastHeard > 20000) load(); });
  load().then(follow);
})(typeof window !== "undefined" ? window : globalThis);
