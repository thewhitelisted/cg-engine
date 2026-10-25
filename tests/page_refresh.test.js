// Live refresh of the public page (league_page.PAGE) in a real DOM (jsdom): changed numbers update in place, an open
// chart stays open and keeps its element, cards re-sort by matchup key, nothing scrolls, the live server is preferred,
// and an older copy never replaces newer numbers. Run by tests/test_page_refresh.py:
//     node tests/page_refresh.test.js A.html B.html B.json OLD.json
const fs = require("fs");
const { JSDOM } = require("jsdom");

const [A, B, BJ, OLDJ] = process.argv.slice(2).map(f => fs.readFileSync(f, "utf8"));
const LIVE = A.match(/const LIVE = "([^"]+)"/)[1];
let failed = 0;
const ok = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) failed++; };
const tick = ms => new Promise(r => setTimeout(r, ms));
const until = async (cond, ms = 5000) => { for (const t0 = Date.now(); !cond() && Date.now() - t0 < ms;) await tick(20); };
const bdoc = new JSDOM(B).window.document;
const keys = d => [...d.querySelectorAll(".wrap .card")].map(c => c.dataset.key);
const bare = el => {                        // markup without script-drawn chart contents or the open flag
  const c = el.cloneNode(true);
  c.querySelectorAll(".mini, .tplot").forEach(x => { x.innerHTML = ""; });
  return c.outerHTML.replace(/ open=""/g, "");
};

function page(serve) {                      // serve(path) -> response body, or null for a network error
  const calls = [];
  const dom = new JSDOM(A, {
    runScripts: "dangerously", pretendToBeVisual: true, url: "https://cg-531d2447.christopherjlee.ca/",
    beforeParse(w) {
      w.IntersectionObserver = class { observe() {} disconnect() {} };
      w.scrollTo = () => calls.push("scrollTo");
      w.fetch = async url => {
        const u = String(url).split("?")[0];
        calls.push(u);
        await tick(5);
        const body = serve(u);
        if (body == null) throw new TypeError("network error");
        return { ok: true, status: 200, json: async () => JSON.parse(body), text: async () => body };
      };
    },
  });
  return { w: dom.window, d: dom.window.document, calls };
}

(async () => {
  {   // the live server has a newer page
    const p = page(u => ({ [LIVE + "data.json"]: BJ, [LIVE + "index.html"]: B }[u] ?? null));
    const det = p.d.querySelector("details.trend");                  // first card's chart, opened by the reader
    const key = det.dataset.key;
    det.open = true;
    const card = det.closest(".card"), nav = p.d.querySelector("nav.jump"), first = p.d.querySelector(".wrap > *");
    await until(() => p.d.body.dataset.v === JSON.parse(BJ).v);
    ok(p.d.body.dataset.v === JSON.parse(BJ).v, "checks the live server on load and takes its newer page");
    ok(p.calls[0] === LIVE + "data.json", "asks the live server first");
    ok(p.d.querySelector(`details.trend[data-key="${key}"]`) === det && det.open, "the open chart is the same element, still open");
    ok(det.querySelector(".tplot").dataset.trend === bdoc.querySelector(`details.trend[data-key="${key}"] .tplot`).dataset.trend,
       "the open chart has the new data");
    ok(det.closest(".card") === card, "its card is the same element");
    ok(JSON.stringify(keys(p.d)) === JSON.stringify(keys(bdoc)) && keys(p.d)[0] !== key, "cards re-sorted to the new order");
    ok(p.d.querySelector("nav.jump") === nav && p.d.querySelector(".wrap > *") === first, "header and section nav kept");
    ok(!p.calls.includes("scrollTo"), "no scrolling");
    ok(bare(p.d.querySelector(".wrap")) === bare(bdoc.querySelector(".wrap")), "markup now matches the new page");
    const n = p.calls.filter(u => u.endsWith("index.html")).length;
    p.d.dispatchEvent(new p.w.Event("visibilitychange"));
    await tick(100);
    ok(p.calls.filter(u => u.endsWith("index.html")).length === n, "same version again: nothing refetched");
    p.w.close();
  }
  {   // live server down; this site's copy is older than the page shown
    const p = page(u => (u === "data.json" ? OLDJ : null));
    const before = bare(p.d.querySelector(".wrap"));
    await until(() => p.calls.includes("data.json"));
    await tick(300);                        // time to (wrongly) apply it
    ok(p.calls.slice(0, 2).join() === `${LIVE}data.json,data.json`, "falls back to the site's own copy");
    ok(bare(p.d.querySelector(".wrap")) === before && p.d.body.dataset.v !== JSON.parse(OLDJ).v, "an older copy is ignored");
    p.w.close();
  }
  {   // live server down; this site's copy is newer
    const p = page(u => ({ "data.json": BJ, "index.html": B }[u] ?? null));
    await until(() => p.d.body.dataset.v === JSON.parse(BJ).v);
    ok(p.d.body.dataset.v === JSON.parse(BJ).v, "a newer site copy is taken when the live server is down");
    p.w.close();
  }
  console.log(failed ? `${failed} FAILED` : "all passed");
  process.exit(failed ? 1 : 0);
})();
