// The players page (players/index.html + web/app.js renderPlayers): every player who has played this week, in the
// order the state gives (best against the model's line first), one row each with his rank, teams, games, vs expected
// and the week's totals; then the real page in a DOM (jsdom, network stubbed) loading the state from the live API,
// following its stream in place, and reading this site's own copy one folder up when the live API doesn't answer.
// Run by tests/test_page_render.py:
//     node tests/players_page.test.js players.html app.js cases.json
const fs = require("fs");
const { JSDOM, VirtualConsole } = require("jsdom");

const [pageFile, appFile, casesFile] = process.argv.slice(2);
const app = require(appFile);
const C = JSON.parse(fs.readFileSync(casesFile, "utf8"));
const LIVE = "https://live.christopherjlee.ca/";
let failed = 0;
const ok = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) failed++; };
const tick = ms => new Promise(r => setTimeout(r, ms));
const until = async (cond, ms = 5000) => { for (const t0 = Date.now(); !cond() && Date.now() - t0 < ms;) await tick(10); };
const parse = html => new JSDOM(`<div class="wrap">${html}</div>`).window.document;
const text = n => n.textContent.trim();
const rows = d => [...d.querySelectorAll("table.players tr")].slice(1);

function checkPage(state, name) {
  const d = parse(app.renderPlayers(state));
  const tr = rows(d), P = state.players;
  ok(tr.length === P.length, `${name}: one row per player (${tr.length})`);
  const bad = [];
  tr.forEach((r, i) => {
    const td = [...r.querySelectorAll("td")].map(text), p = P[i];
    const [pts, reb, ast, stl, blk, tpm, tov, fgm, fga, ftm, fta] = p.s;
    const pc = (m, a) => (a ? app.fixed(m / a, 3).replace(/^0/, "") : "–");
    const want = [String(i + 1), p.player, p.nba, p.team == null ? "—" : p.team, String(p.g),
      p.pct == null ? "—" : `${app.signed(100 * p.pct, 0)}%`, ...[pts, reb, ast, stl, blk, tpm, tov].map(String),
      pc(fgm, fga), pc(ftm, fta)];
    if (td.join("|") !== want.join("|")) bad.push(`row ${i + 1}: ${td.join("|")} != ${want.join("|")}`);
  });
  ok(!bad.length, `${name}: rank, player, teams, games, vs expected, totals and shooting${bad.length ? ": " + bad.slice(0, 3).join("; ") : ""}`);
  ok(text(d.querySelector("h2")) === `${P.length} players`, `${name}: the count`);
  ok(!/playoff|title odds|champion/i.test(d.body.textContent), `${name}: no playoff or title odds`);
}

function open(serve, errors) {
  const calls = [], streams = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", err => errors.push(String(err && (err.detail || err.message || err))));
  const html = fs.readFileSync(pageFile, "utf8")
    .replace('<script src="../app.js"></script>', () => `<script>${fs.readFileSync(appFile, "utf8")}</script>`);
  const dom = new JSDOM(html, {
    runScripts: "dangerously", pretendToBeVisual: true, url: "https://cg-531d2447.christopherjlee.ca/players/", virtualConsole: vc,
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
  ok(text(parse(app.renderPlayers({ ...C.a, players: [] })).querySelector(".note")) === "No games played yet this week.",
     "nobody has played: says so");
  ok(parse(app.renderPlayers({ ...C.a, players: undefined })).body.textContent.includes("No games played yet"),
     "a state from before the players list: no error");
  {
    const errors = [];
    const p = open(u => (u === LIVE + "api/state.json" ? C.a : null), errors);
    await until(() => rows(p.d).length && p.streams.length);
    ok(p.calls[0] === LIVE + "api/state.json", "page: loads the state from the live API");
    ok(p.d.querySelector(".wrap").innerHTML === parse(app.renderPlayers(C.a)).querySelector(".wrap").innerHTML, "and draws the players page");
    const table = p.d.querySelector("table.players");
    p.streams[0].emit("patch", { from: C.a.v, v: C.b.v, ts: C.b.ts, ops: C.ops });
    ok(p.d.querySelector("table.players") === table, "after a patch the table is the same element");
    ok(p.d.querySelector(".wrap").innerHTML === parse(app.renderPlayers(C.b)).querySelector(".wrap").innerHTML, "showing the new list");
    ok(!p.calls.includes("scrollTo"), "no scrolling");
    ok(!errors.length, `no script errors${errors.length ? ": " + errors.join(" | ") : ""}`);
    p.w.close();
  }
  {
    const errors = [];
    const p = open(u => (u === "../api/state.json" ? C.a : null), errors);
    await until(() => rows(p.d).length);
    ok(p.calls.slice(0, 2).join() === `${LIVE}api/state.json,../api/state.json` && rows(p.d).length === C.a.players.length,
       "live API down: reads this site's own copy, one folder up");
    ok(!errors.length, `no script errors${errors.length ? ": " + errors.join(" | ") : ""}`);
    p.w.close();
  }
  console.log(failed ? `${failed} FAILED` : "all passed");
  process.exit(failed ? 1 : 0);
})();
