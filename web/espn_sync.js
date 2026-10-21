// Run in a browser tab signed in to ESPN (fantasy.espn.com). Returns the league snapshot JSON that
// inseason.py reads; save the returned string to data/league_2027.json.
const LEAGUE = 2113064615, SEASON = 2027;
const POS = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"};
const CAT = {0: "PTS", 1: "BLK", 2: "STL", 3: "AST", 6: "REB", 11: "TO", 17: "3PM", 19: "FG%", 20: "FT%"};
const u = `https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/${SEASON}/segments/0/leagues/${LEAGUE}` +
  "?view=mTeam&view=mRoster&view=mMatchupScore&view=mSettings&view=mStatus";
const j = await (await fetch(u, {credentials: "include"})).json();
const side = s => [s.cumulativeScore?.wins ?? 0, s.cumulativeScore?.losses ?? 0, s.cumulativeScore?.ties ?? 0];
const byStat = s => Object.fromEntries(Object.entries(s.cumulativeScore?.scoreByStat || {})
  .filter(([k]) => CAT[k]).map(([k, v]) => [CAT[k], v.result]));
const games = j.schedule.filter(m => m.away);
const decided = games.filter(m => m.winner && m.winner !== "UNDECIDED");
const periods = [...new Set(games.map(m => m.matchupPeriodId))];
JSON.stringify({
  generated: new Date().toISOString(), season: SEASON, league_id: LEAGUE,
  current_period: j.status.currentMatchupPeriod,
  position_limits: Object.fromEntries(Object.entries(j.settings.rosterSettings.positionLimits)
    .filter(([k, v]) => POS[k] && v >= 0).map(([k, v]) => [POS[k], v])),
  order: j.settings.draftSettings.pickOrder,
  teams: j.teams.map(t => ({id: t.id, name: t.name || `${t.location || ""} ${t.nickname || ""}`.trim(),
    record: [t.record?.overall?.wins ?? 0, t.record?.overall?.losses ?? 0, t.record?.overall?.ties ?? 0],
    waiver_rank: t.waiverRank,
    roster: t.roster.entries.map(e => [e.playerId, e.playerPoolEntry.player.fullName, e.lineupSlotId,
      e.playerPoolEntry.player.injuryStatus || ""])})),
  schedule: games.map(m => [m.matchupPeriodId, m.home.teamId, m.away.teamId]),
  results: decided.map(m => [m.matchupPeriodId, m.home.teamId, m.away.teamId, m.winner, ...side(m.home), byStat(m.home)]),
  completed_weeks: periods.filter(p => games.filter(m => m.matchupPeriodId === p).every(m => m.winner && m.winner !== "UNDECIDED")),
});
