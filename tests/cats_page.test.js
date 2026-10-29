// The category page (cats/index.html + web/app.js renderCats): every matchup's chance in each category, with the week's
// totals, drawn from the same state as the league page; then the real page in a DOM (jsdom, network stubbed) loading the
// state from the live API, following its stream in place (cards keep their elements and their fixed order, nothing
// scrolls, nothing throws) and reading this site's own copy one folder up when the live API doesn't answer.
// Run by tests/test_page_render.py:
//     node tests/cats_page.test.js cats.html app.js cases.json
const fs = require("fs");
const { JSDOM, VirtualConsole } = require("jsdom");

const [pageFile, appFile, casesFile] = process.argv.slice(2);
const app = require(appFile);
const C = JSON.parse(fs.readFileSync(casesFile, "utf8"));
const LIVE = "https://live.christopherjlee.ca/";
const CATS = ["FG%", "FT%", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"];
let failed = 0;
const ok = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) failed++; };
const tick = ms => new Promise(r => setTimeout(r, ms));
const until = async (cond, ms = 5000) => { for (const t0 = Date.now(); !cond() && Date.now() - t0 < ms;) await tick(10); };
const parse = html => new JSDOM(`<div class="wrap">${html}</div>`).window.document;
const byKey = (a, b) => String(a.key).localeCompare(String(b.key), "en", { numeric: true });
const keys = d => [...d.querySelectorAll("#matchups > .card")].map(c => c.dataset.key);
const text = n => n.textContent.trim();

function checkPage(state, name) {
  const d = parse(app.renderCats(state));
  const games = [...state.games].sort(byKey);
  ok(JSON.stringify(keys(d)) === JSON.stringify(games.map(g => g.key)), `${name}: one card per matchup, in a fixed order`);
  const links = [...d.querySelectorAll("nav.jump a")].map(a => a.getAttribute("href"));
  ok(JSON.stringify(links) === JSON.stringify(games.map(g => `#m${g.key}`)) && links.every(h => d.getElementById(h.slice(1))),
     `${name}: the matchup links jump to their cards`);
  let bad = [];
  for (const g of games) {
    const card = d.getElementById(`m${g.key}`);
    const pcts = [...card.querySelectorAll(".matchup .pct")].map(text);
    const teams = [...card.querySelectorAll(".matchup .team")].map(text);
    if (pcts.join() !== [app.pct(g.p_home), app.pct(1 - g.p_home)].join() || teams.join() !== [g.home, g.away].join())
      bad.push(`${g.key} header`);
    const rows = [...card.querySelectorAll("table.odds tr")].slice(1);
    if (rows.length !== 9) { bad.push(`${g.key}: ${rows.length} rows`); continue; }
    rows.forEach((tr, i) => {
      const p = g.p_cats[i], [c, h, a] = g.totals[i];
      const td = [...tr.querySelectorAll("td")];
      const bar = [...tr.querySelectorAll(".bar i")].map(x => x.getAttribute("style"));
      const want = [text(tr.querySelector("th")) === c && c === CATS[i], text(td[0]) === h, text(td[4]) === a,
        text(td[1]) === app.pct(p), text(td[3]) === app.pct(1 - p),
        bar.join() === [`width:${app.fixed(100 * p, 1)}%`, `width:${app.fixed(100 * (1 - p), 1)}%`].join(),
        td[0].classList.contains("lead") === (g.lead[i] === 1), td[4].classList.contains("lead") === (g.lead[i] === -1),
        td[1].classList.contains("fav") === (p > 0.5), td[3].classList.contains("fav") === (p < 0.5)];
      if (!want.every(Boolean)) bad.push(`${g.key} ${c} [${want.map(Number).join("")}]`);
    });
    const x = g.p_cats.reduce((s, p) => s + p, 0);
    if (!text(card.querySelector(".caption")).includes(`expected categories ${app.fixed(x, 1)}–${app.fixed(9 - x, 1)}`))
      bad.push(`${g.key} expected`);
  }
  ok(!bad.length, `${name}: names, week chances, each category's chances, bars, totals and leaders${bad.length ? ": " + bad.join("; ") : ""}`);
  ok(!/playoff|title odds|champion/i.test(d.body.textContent), `${name}: no playoff or title odds`);
}

function open(serve, errors) {             // serve(url) -> state object, or null for a network error
  const calls = [], streams = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", err => errors.push(String(err && (err.detail || err.message || err))));
  const html = fs.readFileSync(pageFile, "utf8")
    .replace('<script src="../app.js"></script>', () => `<script>${fs.readFileSync(appFile, "utf8")}</script>`);
  const dom = new JSDOM(html, {
    runScripts: "dangerously", pretendToBeVisual: true, url: "https://cg-531d2447.christopherjlee.ca/cats/", virtualConsole: vc,
    beforeParse(w) {
      w.scrollTo = () => calls.push("scrollTo");
      w.fetch = async url => {
        const u = String(url).split("?")[0];
        calls.push(u);
        await tick(2);
        const body = serve(u);
        if (body == null) throw new TypeError("network error");
        return { ok: true, status: 200, json: async () => JSON.parse(JSON.stringify(body)) };
      };
      w.EventSource = class {
        constructor(url) { this.url = url; this.on = {}; streams.push(this); }
        addEventListener(t, f) { this.on[t] = f; }
        emit(t, data) { this.on[t] && this.on[t]({ data: JSON.stringify(data) }); }
      };
    },
  });
  return { w: dom.window, d: dom.window.document, calls, streams };
}

(async () => {
  checkPage(C.a, "live state");
  checkPage(C.b, "after a change");
  {
    const d = parse(app.renderCats(C.old));
    const odds = [...d.querySelectorAll("table.odds tr td.p")];
    ok(keys(d).length === C.old.games.length && odds.length === 9 * C.old.games.length && odds.every(td => text(td) === "—") &&
       !d.body.textContent.includes("expected categories"), "a state without category chances: totals and dashes, no error");
    ok(parse(app.renderCats({ ...C.a, games: [] })).body.textContent.includes("No matchups this week."), "no matchups: says so");
  }
  {
    const errors = [];
    let current = C.a;
    const p = open(u => (u === LIVE + "api/state.json" ? current : null), errors);
    await until(() => keys(p.d).length && p.streams.length);
    ok(p.calls[0] === LIVE + "api/state.json", "page: loads the state from the live API");
    ok(p.d.querySelector(".wrap").innerHTML === parse(app.renderCats(C.a)).querySelector(".wrap").innerHTML, "and draws the category page");
    ok(p.streams[0].url === `${LIVE}api/events?v=${encodeURIComponent(C.a.v)}`, "then follows the live event stream");
    const card = p.d.getElementById(`m${C.key}`), row = card.querySelectorAll("table.odds tr")[4], order = keys(p.d);
    p.streams[0].emit("patch", { from: C.a.v, v: C.b.v, ts: C.b.ts, ops: C.ops });
    ok(p.d.getElementById(`m${C.key}`) === card && card.querySelectorAll("table.odds tr")[4] === row,
       "after a patch the card and its rows are the same elements");
    ok(p.d.querySelector(".wrap").innerHTML === parse(app.renderCats(C.b)).querySelector(".wrap").innerHTML, "showing the new numbers");
    ok(JSON.stringify(keys(p.d)) === JSON.stringify(order), "cards keep their order when the league page's order changes");
    ok(!p.calls.includes("scrollTo"), "no scrolling");
    ok(!errors.length, `no script errors${errors.length ? ": " + errors.join(" | ") : ""}`);
    p.w.close();
  }
  {
    const errors = [];
    const p = open(u => (u === "../api/state.json" ? C.a : null), errors);
    await until(() => keys(p.d).length);
    ok(p.calls.slice(0, 2).join() === `${LIVE}api/state.json,../api/state.json` && keys(p.d).length === C.a.games.length,
       "live API down: reads this site's own copy, one folder up");
    ok(!errors.length, `no script errors${errors.length ? ": " + errors.join(" | ") : ""}`);
    p.w.close();
  }
  console.log(failed ? `${failed} FAILED` : "all passed");
  process.exit(failed ? 1 : 0);
})();
