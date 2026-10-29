// web/app.js render(state) must reproduce, character for character, the page the original Python renderer produced for
// the same data (tests/fixtures/page_cases.json.gz: real data plus synthetic states covering every section and edge),
// except for changes made since on purpose: a tied week in "odds against results" carries no ✓ / (upset) mark.
//     node tests/page_render.test.js [CASES.json.gz]
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const app = require(path.join(__dirname, "..", "web", "app.js"));

const file = process.argv[2] || path.join(__dirname, "fixtures", "page_cases.json.gz");
const cases = JSON.parse(zlib.gunzipSync(fs.readFileSync(file)).toString("utf8"));
let failed = 0;
for (const c of cases) {
  const got = app.render(c.state);
  if (got === c.html) { console.log(`PASS ${c.name} (${got.length} chars)`); continue; }
  failed++;
  let i = 0;
  while (i < got.length && got[i] === c.html[i]) i++;
  console.log(`FAIL ${c.name}: first difference at ${i}\n  want …${JSON.stringify(c.html.slice(Math.max(0, i - 80), i + 80))}…` +
              `\n  got  …${JSON.stringify(got.slice(Math.max(0, i - 80), i + 80))}…`);
}
// Python's number formatting, including exact halves (rounded to even) and negative zero
const fmt = [[app.fixed(0.0625, 3), "0.062"], [app.fixed(0.1875, 3), "0.188"], [app.pct(0.125), "12%"], [app.pct(0.875), "88%"],
  [app.pct(0.995), "100%"], [app.fixed(-0.04, 1), "-0.0"], [app.signed(0, 1), "+0.0"], [app.signed(-0.4, 0), "-0"],
  [app.fixed(2.5, 0), "2"], [app.fixed(3.5, 0), "4"], [app.g6(1.5), "1.5"], [app.g6(2.0), "2"]];
for (const [got, want] of fmt) {
  if (got !== want) { failed++; console.log(`FAIL format: got ${got}, want ${want}`); }
}
// a tied week is neither the favourite's win nor an upset
const tied = cases.find(c => c.state.last && c.state.last.rows.some(r => r.tie && r.p_home != null));
const row = tied && app.render(tied.state).match(/<tr><td>[^<]* vs [^<]*<\/td>(?:(?!<\/tr>).)*>Tie[^<]*<\/td><\/tr>/);
if (!row || /✓|upset/.test(row[0])) { failed++; console.log(`FAIL tie mark: ${row ? row[0] : "no tied row with odds"}`); }
// the players page button sits under the best and worst players, and only when there are some
const bare = app.render({ ...cases[0].state, top: [], worst: [] });
if (bare.includes('href="players/"')) { failed++; console.log("FAIL players button shown without best/worst players"); }
if (!/<section id="worst"[^]*?<\/section>\n  <div class="more"><a class="btn" href="players\/">/.test(app.render(cases[0].state))) {
  failed++; console.log("FAIL players button not right under the worst players");
}
console.log(failed ? `${failed} FAILED` : `all ${cases.length} pages identical; formatting, tie and players-button checks passed`);
process.exit(failed ? 1 : 0);
