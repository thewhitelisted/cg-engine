# cg-engine

Private engine for the Cool Guys league odds page. A scheduled GitHub Actions workflow
(`.github/workflows/publish.yml`) runs `publish_league.py` every 30 minutes: it pulls the league from ESPN
(cookies in the `ESPN_S2` / `ESPN_SWID` repository secrets), computes this week's matchup odds with the
live model, and pushes the page to the public `cg-531d2447` Pages repo (token in `PAGES_TOKEN`).
Run it by hand from the Actions tab ("Run workflow").
