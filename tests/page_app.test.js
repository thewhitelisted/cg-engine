// The real page shell + web/app.js in a DOM (jsdom), with the network stubbed: the page loads the state from the live API,
// follows its event stream, applies patches in place (an open chart stays open and keeps its element, cards re-sort by
// matchup, nothing scrolls), fetches everything when it falls behind, ignores older states, and reads this site's own
// copy when the live API doesn't answer. Run by tests/test_page_render.py:
//     node tests/page_app.test.js shell.html app.js cases.json
const fs = require("fs");
const { JSDOM } = require("jsdom");

const [shellFile, appFile, casesFile] = process.argv.slice(2);
const app = require(appFile);
const C = JSON.parse(fs.readFileSync(casesFile, "utf8"));
const LIVE = "https://live.christopherjlee.ca/";
const page = fs.readFileSync(shellFile, "utf8")
  .replace('<script src="app.js"></script>', () => `<script>${fs.readFileSync(appFile, "utf8")}</script>`);
let failed = 0;
const ok = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) failed++; };
const tick = ms => new Promise(r => setTimeout(r, ms));
const until = async (cond, ms = 5000) => { for (const t0 = Date.now(); !cond() && Date.now() - t0 < ms;) await tick(10); };
const bare = html => {                     // markup without script-drawn chart contents or the open flag
  const d = new JSDOM(`<div class="wrap">${html}</div>`).window.document;
  d.querySelectorAll(".mini, .tplot").forEach(x => { x.innerHTML = ""; });
  return d.querySelector(".wrap").innerHTML.replace(/ open=""/g, "");
};
const keys = d => [...d.querySelectorAll("#matchups > .card")].map(c => c.dataset.key);

function open(serve) {                     // serve(url) -> state object, or null for a network error
  const calls = [], streams = [];
  const dom = new JSDOM(page, {
    runScripts: "dangerously", pretendToBeVisual: true, url: "https://cg-531d2447.christopherjlee.ca/",
    beforeParse(w) {
      w.IntersectionObserver = class { observe() {} disconnect() {} };
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
  {
    let current = C.a;
    const p = open(u => (u === LIVE + "api/state.json" ? current : null));
    ok(p.d.head.innerHTML.indexOf('http-equiv="refresh"') < 0, "no timed full reloads");
    await until(() => keys(p.d).length && p.streams.length);
    ok(p.calls[0] === LIVE + "api/state.json", "loads the state from the live API");
    ok(bare(p.d.querySelector(".wrap").innerHTML) === bare(app.render(C.a)), "draws the page from it");
    ok(p.streams[0].url === `${LIVE}api/events?v=${encodeURIComponent(C.a.v)}`, "then follows the live event stream");
    const det = p.d.querySelector(`details.trend[data-key="${C.key}"]`);
    det.open = true;
    const card = det.closest(".card"), nav = p.d.querySelector("nav.jump");
    p.streams[0].emit("patch", { from: C.a.v, v: C.b.v, ts: C.b.ts, ops: C.ops });
    ok(p.d.querySelector(`details.trend[data-key="${C.key}"]`) === det && det.open, "after a patch the open chart is the same element, still open");
    ok(det.querySelector(".tplot").dataset.trend === new JSDOM(app.render(C.b)).window.document
      .querySelector(`details.trend[data-key="${C.key}"] .tplot`).dataset.trend, "and shows the new data");
    ok(det.closest(".card") === card && p.d.querySelector("nav.jump") === nav, "its card and the section nav are kept");
    ok(JSON.stringify(keys(p.d)) === JSON.stringify(C.b.games.map(g => g.key)), "cards re-sorted to the new order");
    ok(bare(p.d.querySelector(".wrap").innerHTML) === bare(app.render(C.b)), "the page now matches the new state");
    ok(!p.calls.includes("scrollTo"), "no scrolling");
    const n = p.calls.length;
    p.streams[0].emit("patch", { from: C.b.v, v: C.b.v, ts: C.b.ts, ops: [] });
    await tick(30);
    ok(p.calls.length === n, "a patch it already has is ignored");
    current = C.b;
    p.streams[0].emit("patch", { from: "something-else", v: "zzzz", ts: C.b.ts + 5, ops: [] });
    await until(() => p.calls.length > n);
    ok(p.calls[n] === LIVE + "api/state.json", "a patch from a version it doesn't have: fetches the whole state");
    p.streams[0].emit("state", C.older);
    await tick(30);
    ok(p.d.querySelector(".sub").textContent.includes(C.b.updated), "an older whole state is ignored");
    p.w.close();
  }
  {
    const p = open(u => (u === "api/state.json" ? C.a : null));
    await until(() => keys(p.d).length);
    ok(p.calls.slice(0, 2).join() === `${LIVE}api/state.json,api/state.json` && keys(p.d).length === C.a.games.length,
       "live API down: reads this site's own copy");
    p.w.close();
  }
  console.log(failed ? `${failed} FAILED` : "all passed");
  process.exit(failed ? 1 : 0);
})();
