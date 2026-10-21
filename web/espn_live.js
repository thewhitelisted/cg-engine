// Run in a browser tab signed in to ESPN. Returns live matchup totals for the current period (data/live.json).
const LEAGUE = 2113064615, SEASON = 2027;
const ST = {0: "pts", 1: "blk", 2: "stl", 3: "ast", 6: "reb", 11: "tov", 13: "fgm", 14: "fga", 15: "ftm", 16: "fta", 17: "tpm"};
const u = `https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/${SEASON}/segments/0/leagues/${LEAGUE}?view=mMatchupScore&view=mScoreboard&view=mStatus`;
const j = await (await fetch(u, {credentials: "include"})).json();
const cur = j.status.currentMatchupPeriod;
const tot = s => Object.fromEntries(Object.entries((s.cumulativeScoreLive || s.cumulativeScore || {}).scoreByStat || {})
  .filter(([k]) => ST[k]).map(([k, v]) => [ST[k], v.score || 0]));
JSON.stringify({generated: new Date().toISOString(), period: cur, latest_scoring_period: j.status.latestScoringPeriod,
  matchups: j.schedule.filter(m => m.matchupPeriodId === cur && m.away).map(m => ({home: m.home.teamId, away: m.away.teamId, home_tot: tot(m.home), away_tot: tot(m.away)}))});
