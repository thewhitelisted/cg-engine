"""Season context, player table and H2H 9-cat matchup model.

`Season(t)` rebuilds everything as of ESPN season id t (2027 = 2026-27) using only data from
before that season: projections (projections.py, fit on seasons < t), projection-error model,
fantasy-playoff rest effect, schedule and the draft market (ADP).

Weekly matchup model: a team's weekly total in each raw stat is a sum over its players of
games * per-game rate. Per-game variances follow var = a * mean^b (fit on game logs); weekly games
played vary (var ~ GAMES_VAR_RATIO * mean, teammates correlated via the shared schedule).
Categories are compared with a normal approximation; FG%/FT% via linearised excess makes.
Validated against real 2025-26 weeks in calib.py.

Projection uncertainty is handled two ways:
  * analytically (extra variance) for fast paths (category weights, the CLI board), and
  * by Monte Carlo "true season" draws in sim.Draft.evaluate, which also lets busts be dropped
    for waiver players (option value of risky picks).
"""
import json
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import projections as Pj

DATA = Path(__file__).parent / "data"

CATS = ["FG%", "FT%", "3PM", "REB", "AST", "STL", "BLK", "TO", "PTS"]
STATK = ["fgx", "fga", "ftx", "fta", "tpm", "reb", "ast", "stl", "blk", "tov", "pts"]
SK = {s: i for i, s in enumerate(STATK)}
N_TEAMS, ROSTER = 12, 13
LIVE_SEASON = 2027

LG_FG, LG_FT = 0.4716, 0.7865
# per-game variance = a * base_mean ** b, fit on 2025-26 game logs (players with >=25 games)
VAR_FIT = {"pts": ("pts", 4.704, 0.842), "reb": ("reb", 1.400, 0.989), "ast": ("ast", 1.260, 0.990),
           "stl": ("stl", 1.068, 0.961), "blk": ("blk", 1.152, 1.097), "tov": ("tov", 1.050, 0.923),
           "tpm": ("tpm", 0.992, 1.531), "fgx": ("fga", 0.261, 0.937), "ftx": ("fta", 0.180, 0.963)}
# Week-to-week games played: var(games) ~= 0.46 * mean(games) (2025-26 logs; see calib.py).
GAMES_VAR_RATIO = 0.46
# Correlation of teammates' weekly games played (shared schedule, team-wide rest): 0.19-0.30 measured.
TEAMMATE_GAMES_RHO = 0.25
# Fraction of an injured player's missed games recovered via IR + streaming a waiver player.
INJURY_RECOVERY = 0.5
# In-season waivers: a bust is identified and replaced after this fraction of the season.
LEARN_FRAC = 0.2
WAIVER_DEPTH = 12        # replacement line = mean of the best this-many undrafted players
PLAYOFF_START = "03-08"  # fantasy playoffs: 3 Mon-Sun weeks starting the first Monday on/after Mar 8
REST_BASE = 0.777        # participation outside the playoff window (research_rest.py)
# Projection-error draws: "gauss" (v1/v2: multivariate normal) or "boot" (candidate 6: joint bootstrap of
# standardized residual vectors + GP residuals from the same predicted-share tercile).
ERROR_DRAWS = "gauss"


def set_error_draws(kind: str):
    global ERROR_DRAWS
    ERROR_DRAWS = kind


# Candidate 8: shrink model value toward the market's view (ADP) by MARKET_BETA, via a per-player role
# multiplier. Applied only in seasons with real preseason ADP. 0 reproduces v1/v2.
MARKET_BETA = 0.0
REAL_ADP_SEASONS = {2022, 2024, 2026, 2027}
# Seasons whose market comes from archived preseason ESPN ADP rank (archive_adp.py)
ARCHIVE_ADP_SEASONS = {2026}


def set_market_beta(b: float):
    global MARKET_BETA
    MARKET_BETA = b
    get_season.cache_clear()


def market_adjust(df: pd.DataFrame, beta: float, real_only: bool = False) -> pd.DataFrame:
    """Move each player's value toward the value at his market rank, scaling counting stats/attempts.
    real_only (candidate 10): only players with a real ADP (market rank <= 135) are adjusted."""
    if beta == 0:
        return df
    val, scale = static_value(df)
    mrank = df["mkt"].rank(method="first").astype(int).to_numpy()
    mkt_val = np.sort(val)[::-1][np.clip(mrank - 1, 0, len(val) - 1)]
    target = val + beta * (mkt_val - val)
    mu = df[STATK].to_numpy(float)
    av = df["avail"].to_numpy()
    A = sum(sign * mu[:, j] * av / sd for j, m0, sd, sign in scale.values())
    B = sum(sign * m0 / sd for j, m0, sd, sign in scale.values())
    m = np.where(np.abs(A) > 1e-6, (target + B) / np.where(np.abs(A) > 1e-6, A, 1), 1.0)
    m = np.clip(m, 0.5, 1.6)
    if real_only:
        m = np.where(df["mkt"].to_numpy() <= 135, m, 1.0)
    out = df.copy()
    for k in STATK + ["fgm", "ftm"]:
        out[k] = df[k] * m
    out["market_mult"] = m
    return out


# Weekly matchup sd multiplier, fit on 2024-25 real weeks and validated on 2025-26 (research_calib.py).
# Set via set_kappa(); 1.0 reproduces v1.
MATCHUP_SD_KAPPA = 1.0


def set_kappa(k: float):
    global MATCHUP_SD_KAPPA
    MATCHUP_SD_KAPPA = k
# Seasons whose stored ESPN ADP is genuinely preseason (2025 and 2026 files were updated in-season)
CLEAN_ADP_SEASONS = {2022, 2024, 2027}


@lru_cache(maxsize=8)
def get_season(t: int = LIVE_SEASON, proj: str = "stack") -> "Season":
    return Season(t, proj)


def load_players(t: int = LIVE_SEASON) -> pd.DataFrame:
    return get_season(t).df


class Season:
    def __init__(self, t: int, proj: str = "stack"):
        self.t, self.proj = t, proj
        self.games = Pj.sgames(t)
        self.err = Pj.error_params(range(Pj.train_start(t), t))
        self._build_schedule()
        self.df = self._build_players()

    # ------------------------------------------------------------------
    def _build_players(self) -> pd.DataFrame:
        t = self.t
        info = Pj.files()[t][["name", "pro_team_id", "pos", "elig", "injury", "adp", "espn_rank"]]
        pr = Pj.project(t)[0] if self.proj.startswith("stack") else Pj.project_espn(t)
        espn = Pj.espn_proj(t)
        df = info.join(pr, how="inner")
        df["espn_gp"] = espn["gp"].reindex(df.index).fillna(0)
        df["espn_min"] = espn["min"].reindex(df.index).fillna(0)
        df = df[df["espn_min"] >= 8].copy()
        df["avail"] = df["gp_share"].clip(0, 1)
        df["gp"] = df["avail"] * self.games
        df["fgx"] = df["fgm"] - LG_FG * df["fga"]
        df["ftx"] = df["ftm"] - LG_FT * df["fta"]
        teams = pd.read_csv(DATA / "teams.csv").set_index("pro_team_id")["abbrev"]
        df["team"] = df["pro_team_id"].map(teams).fillna("FA")

        # draft market (what opponents draft from)
        if t in ARCHIVE_ADP_SEASONS:
            arch = pd.read_csv(DATA / f"adp_archive_{t}.csv").set_index("id")["adp_espn"]
            e = espn.reindex(df.index)
            tmp = df.copy()
            for k in Pj.STATS:
                tmp[k] = e[k]
            tmp["fgx"] = tmp["fgm"] - LG_FG * tmp["fga"]
            tmp["ftx"] = tmp["ftm"] - LG_FT * tmp["fta"]
            tmp["avail"] = (e["gp"] / self.games).clip(0, 1)
            proxy = pd.Series(static_value(tmp, pool_by="value")[0], index=df.index).rank(ascending=False)
            a = arch.reindex(df.index)
            rest = proxy[a.isna()].rank() + a.max()
            df["mkt"] = a.fillna(rest)
        elif t in CLEAN_ADP_SEASONS:
            df["mkt"] = np.where(df["adp"] < 135, df["adp"], np.maximum(df["espn_rank"].fillna(400), 135))
        else:
            # 2024-25 / 2025-26: the stored ADP and ranks include in-season updates (e.g. Chet Holmgren
            # rank 142 after his injury), which would let simulated opponents dodge busts with hindsight.
            # Proxy the market with the rank of ESPN's own preseason projections (clean).
            e = espn.reindex(df.index)
            tmp = df.copy()
            for k in Pj.STATS:
                tmp[k] = e[k]
            tmp["fgx"] = tmp["fgm"] - LG_FG * tmp["fga"]
            tmp["ftx"] = tmp["ftm"] - LG_FT * tmp["fta"]
            tmp["avail"] = (e["gp"] / self.games).clip(0, 1)
            v = static_value(tmp, pool_by="value")[0]
            df["mkt"] = pd.Series(v, index=df.index).rank(ascending=False)
        df["mkt"] = df["mkt"].astype(float)

        # fantasy-playoff participation multiplier from last season's team record
        rec = pd.read_csv(DATA / "team_records.csv")
        wp = rec[rec.season == t - 1].set_index("pro_team_id")["win_pct"]
        df["wp_prev"] = df["pro_team_id"].map(wp).fillna(0.5)
        rw = pd.read_csv(DATA / "rest_window.csv")
        rw = rw[rw.season != t]  # out-of-sample for historical seasons
        a, b = np.linalg.lstsq(np.c_[np.ones(len(rw)), rw.wp_prev], rw["drop"], rcond=None)[0]
        df["po_mult"] = np.clip(1 + (a + b * df["wp_prev"]) / REST_BASE, 0.5, 1.05)

        beta = MARKET_BETA
        if "_mkt" in self.proj:  # candidate 8 variant: beta fit on 2023-24 (research_market.py)
            beta = json.loads((DATA / "market_beta.json").read_text())["beta"]
        if self.t in REAL_ADP_SEASONS and beta != 0:
            df = market_adjust(df, beta, real_only=self.proj.endswith("_mkt10"))
        df = df.sort_values("mkt").reset_index()
        if self.proj == "stack_belief":
            df = self._apply_belief(df)
        return df

    def _apply_belief(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shipped belief layer (belief.py, V2): center = stack shifted by the predictive-mean market weights;
        the season-level uncertainty of those weights is kept in self.belief for the simulation engine."""
        import belief as Bf

        h = Bf.load_for_season(self.t)
        val, scale = static_value(df)
        Dm = Bf.disagreement_matrix(df, val, self.t, h["variant"])          # [K, P]
        mu = df[STATK].to_numpy(float)
        av = df["avail"].to_numpy()
        A = sum(sign * mu[:, j] * av / sd for j, m0, sd, sign in scale.values())
        if h["variant"] in Bf.HIER:
            hy = {"variant": h["variant"], **{k: np.atleast_2d(np.asarray(h[k])) for k in
                                              ("beta_mu", "beta_tau", "gamma_mu", "gamma_tau")}}
        elif h["variant"] == "V1":
            hy = {"variant": "V2", "beta_mu": np.array([[h["beta"]]]), "beta_tau": np.array([[0.0]]),
                  "gamma_mu": np.array([[0.0]]), "gamma_tau": np.array([[0.0]])}
        else:
            hy = {"variant": "V2", **{k: np.array([[0.0]]) for k in ("beta_mu", "beta_tau", "gamma_mu", "gamma_tau")}}
        bbar, gbar = hy["beta_mu"].mean(1), hy["gamma_mu"].mean(1)               # [K]
        mbar = Bf.mult_from_shift(bbar @ Dm, A)
        gshift = gbar @ Dm
        # spread of the season-level market adjustment, for the analytic path
        bs, gs = Bf.market_draws(hy, 4000, np.random.default_rng(123))
        m_sd = (Bf.mult_from_shift(bs.T @ Dm, A) / np.maximum(mbar, 1e-6)).std(0)
        g_sd = (gs.T @ Dm).std(0)
        out = df.copy()
        for k in STATK + ["fgm", "ftm"]:
            out[k] = df[k] * mbar
        out["avail"] = np.clip(av + gshift, 0.0, 1.0)
        out["gp"] = out["avail"] * self.games
        out["belief_mult"] = mbar
        self.belief = {"hy": hy, "Dm": Dm, "A": A, "mbar": mbar, "gshift": gshift, "m_sd": m_sd, "g_sd": g_sd,
                       "s_v": float(h.get("s_v", 1.0))}
        return out

    def _build_schedule(self):
        s = pd.read_csv(DATA / f"schedule_{self.t}.csv", parse_dates=["date"])
        first = s["date"].min()
        self.season_start = first - pd.Timedelta(days=first.weekday())
        s["wk"] = (s["date"] - self.season_start).dt.days // 7
        self.wg = s.groupby(["pro_team_id", "wk"]).size().unstack(fill_value=0)
        po = pd.Timestamp(f"{self.t}-{PLAYOFF_START}")
        po = po + pd.Timedelta(days=(7 - po.weekday()) % 7)
        w0 = (po - self.season_start).days // 7
        self.playoff_weeks = (w0, w0 + 1, w0 + 2)
        self.reg_weeks = [w for w in self.wg.columns if w < w0 and self.wg[w].sum() > 80]
        self.playoff_dates = (po, po + pd.Timedelta(days=20))

    # ------------------------------------------------------------------
    @cached_property
    def games_matrix(self) -> np.ndarray:
        """Scheduled games per player for [avg reg week, playoff wk 1, 2, 3], before availability."""
        cols = [self.wg[self.reg_weeks].mean(axis=1)] + [self.wg[w] for w in self.playoff_weeks]
        g = np.stack([self.df["pro_team_id"].map(c).fillna(0).to_numpy(float) for c in cols], axis=1)
        g[:, 1:] *= self.df["po_mult"].to_numpy()[:, None]
        return g

    @cached_property
    def team_idx(self) -> np.ndarray:
        return self.df["pro_team_id"].fillna(0).astype(int).to_numpy()


def per_game_var(mu: np.ndarray) -> np.ndarray:
    """Per-game variance for STATK given per-game means [..., 11]."""
    var = np.zeros_like(mu)
    for s, (base, a, b) in VAR_FIT.items():
        var[..., SK[s]] = a * np.maximum(mu[..., SK[base]], 0.01) ** b
    return var


class Evaluator:
    """Weekly category win probabilities between rosters, vectorised over many rosters."""

    def __init__(self, season: Season, proj_unc: bool = True):
        self.season = season
        df = season.df
        self.df = df
        self.mu = df[STATK].to_numpy(float)
        self.var = per_game_var(self.mu)
        self.g = season.games_matrix
        self.avail = df["avail"].to_numpy()
        self.team = season.team_idx
        val, self.vscale = static_value(df)
        order = np.argsort(-val)
        lo, hi = N_TEAMS * ROSTER, N_TEAMS * ROSTER + 45
        self.repl_mu = self.mu[order[lo:hi]].mean(axis=0)
        self.repl_var = self.var[order[lo:hi]].mean(axis=0)
        self.wmu, self.wvar, self.own = self.weekly(self.mu, self.avail)
        self.bel = getattr(season, "belief", None)
        s_v = self.bel["s_v"] if self.bel else 1.0
        if proj_unc:
            # Analytic average over projection error: add its variance to the weekly-mean variance.
            pv = projection_var(df, season.err) * s_v**2
            a = self.avail[:, None, None]
            gp_sd = season.err["gp_resid"].std() * s_v
            gap = self.mu[:, None, :] - INJURY_RECOVERY * self.repl_mu[None, None, :]
            extra = a**2 * pv[:, None, :] + (gp_sd * gap) ** 2
            if self.bel:
                # uncertainty in the season's market weights (belief layer): rate multiplier and GP shift
                b = self.bel
                extra = extra + a**2 * (b["m_sd"][:, None, None] * self.mu[:, None, :]) ** 2
                extra = extra + (b["g_sd"][:, None, None] * gap) ** 2
            self.wvar = self.wvar + self.g[:, :, None] ** 2 * extra

    def weekly(self, mu, avail, g=None):
        """Weekly mean/var contributions [..., week, stat] from per-game mu [..., stat] and avail [...].
        g: scheduled games [..., week] (defaults to all players)."""
        g = self.g if g is None else g
        own = g * avail[..., None]
        rec = g * (1 - avail[..., None]) * INJURY_RECOVERY
        var = per_game_var(mu)
        wmu = own[..., None] * mu[..., None, :] + rec[..., None] * self.repl_mu
        wvar = (own[..., None] * var[..., None, :] + rec[..., None] * self.repl_var
                + GAMES_VAR_RATIO * own[..., None] * mu[..., None, :] ** 2)
        return wmu, wvar, own

    def _sum_team(self, m, v, o, pm, rosters, g=None):
        """Sum gathered per-slot arrays [..., n, W, S] into team totals, adding teammate covariance."""
        idx = np.where(rosters < 0, 0, rosters)
        empty = (rosters == -1)[..., None, None]          # empty active slot: replacement-level filler
        void = (rosters <= -2)[..., None, None]           # unused slot (e.g. empty IR): contributes nothing
        g_avg = (self.g if g is None else g).mean(axis=0)[:, None]
        m = np.where(empty, g_avg * self.repl_mu, np.where(void, 0.0, m))
        v = np.where(empty, g_avg * self.repl_var, np.where(void, 0.0, v))
        # teammates' games co-move: Var += rho * sum_{i!=j same team} sd_i sd_j, sd_i = sqrt(GV*own_i)*mu_i
        a = np.where(empty | void, 0, np.sqrt(GAMES_VAR_RATIO * o)[..., None] * pm[..., None, :])
        tm = np.where(rosters < 0, -1, self.team[idx])
        onehot = (tm[..., None] == np.arange(31)).astype(float)
        S = np.einsum("...nk,...nws->...kws", onehot, a)
        cross = (S**2).sum(axis=-3) - (a**2).sum(axis=-3)
        return m.sum(axis=-3), v.sum(axis=-3) + TEAMMATE_GAMES_RHO * cross

    def team_stats(self, rosters):
        """Projected team weekly totals [..., week, stat] for rosters [..., n] (-1 = replacement filler)."""
        idx = np.where(rosters < 0, 0, rosters)
        return self._sum_team(self.wmu[idx], self.wvar[idx], self.own[idx], self.mu[idx], rosters)

    def truth_team_stats(self, truth: dict, rosters: np.ndarray, waivers: bool = True, g=None):
        """Team weekly totals under per-rollout true seasons. rosters [R, T, n] (all teams).

        With waivers, each drafted bust (true value below the waiver pool) is replaced by the waiver
        line for (1 - LEARN_FRAC) of the season. g: scheduled games [P, W] (default: typical week + playoffs).
        """
        R = rosters.shape[0]
        idx = np.where(rosters < 0, 0, rosters)
        r = np.arange(R)[:, None, None]
        mu, avail = truth["mu"][r, idx], truth["avail"][r, idx]          # [R, T, n, S], [R, T, n]
        if waivers:
            value = truth["value"]
            r1 = np.arange(R)[:, None]
            drafted = np.zeros(value.shape, bool)
            drafted[r1, idx.reshape(R, -1)] = True
            fa_val = np.where(drafted, -np.inf, value)
            top = np.argpartition(-fa_val, WAIVER_DEPTH, axis=1)[:, :WAIVER_DEPTH]
            rep_mu = truth["mu"][r1, top].mean(1)[:, None, None, :]
            rep_avail = truth["avail"][r1, top].mean(1)[:, None, None]
            rep_val = value[r1, top].mean(1)[:, None, None]
            bust = (value[r, idx] < rep_val) & (rosters >= 0)
            mu = np.where(bust[..., None], LEARN_FRAC * mu + (1 - LEARN_FRAC) * rep_mu, mu)
            avail = np.where(bust, LEARN_FRAC * avail + (1 - LEARN_FRAC) * rep_avail, avail)
        G = self.g if g is None else g
        wmu, wvar, own = self.weekly(mu, avail, G[idx])
        return self._sum_team(wmu, wvar, own, mu, rosters, G)

    @staticmethod
    def cat_probs(mA, vA, mB, vB):
        """P(A beats B) in each of the 9 categories. Inputs [..., stat]; output [..., 9]."""
        k2 = MATCHUP_SD_KAPPA**2
        vA, vB = vA * k2, vB * k2
        out = []
        for x, a in (("fgx", "fga"), ("ftx", "fta")):
            ra = mA[..., SK[x]] / mA[..., SK[a]]
            rb = mB[..., SK[x]] / mB[..., SK[a]]
            sd = np.sqrt(vA[..., SK[x]] / mA[..., SK[a]] ** 2 + vB[..., SK[x]] / mB[..., SK[a]] ** 2)
            out.append(norm.cdf((ra - rb) / sd))
        for s in ("tpm", "reb", "ast", "stl", "blk", "tov", "pts"):
            p = norm.cdf((mA[..., SK[s]] - mB[..., SK[s]]) / np.sqrt(vA[..., SK[s]] + vB[..., SK[s]]))
            out.append(1 - p if s == "tov" else p)
        return np.stack(out, axis=-1)

    @staticmethod
    def matchup_win(p):
        """P(win more categories than opponent) from per-cat win probs [..., 9] (Poisson-binomial)."""
        dist = np.zeros(p.shape[:-1] + (10,))
        dist[..., 0] = 1.0
        for c in range(9):
            pc = p[..., c : c + 1]
            dist = np.concatenate([dist[..., :1] * (1 - pc), dist[..., 1:] * (1 - pc) + dist[..., :-1] * pc], axis=-1)
        return dist[..., 5:].sum(axis=-1)

    # ---- Monte Carlo truth -------------------------------------------
    def draw_truth(self, R: int, rng) -> dict:
        """R simulated true seasons: per-game rates [R, P, 11] and availability [R, P]."""
        err = self.season.err
        df = self.df
        P = len(df)
        s_v = self.bel["s_v"] if self.bel else 1.0
        if ERROR_DRAWS == "boot":
            z = err["z_rows"][rng.integers(0, len(err["z_rows"]), (R, P))]
        else:
            L = np.linalg.cholesky(err["corr"] + 1e-9 * np.eye(len(Pj.ERR_STATS)))
            z = rng.standard_normal((R, P, len(Pj.ERR_STATS))) @ L.T
        z = z * s_v
        # in-season: remaining uncertainty about a player's true rate shrinks as games accumulate
        # (season.err_mult = sqrt(K / (K + n)) per player; None before the season)
        em = getattr(self.season, "err_mult", None)
        em = np.ones(P) if em is None else np.asarray(em, float)
        z = z * em[None, :, None]
        raw = {}
        for j, k in enumerate(Pj.ERR_STATS):
            f = df[k].to_numpy()
            raw[k] = np.maximum(f + err["scale"][k] * np.sqrt(np.maximum(f, 0.05)) * z[..., j], 0)
        fgp = (df["fgm"] / df["fga"].clip(lower=0.1)).to_numpy() + err["fg_sd"] * s_v * em * rng.standard_normal((R, P))
        ftp = (df["ftm"] / df["fta"].clip(lower=0.1)).to_numpy() + err["ft_sd"] * s_v * em * rng.standard_normal((R, P))
        mu = np.zeros((R, P, len(STATK)))
        mu[..., SK["fga"]], mu[..., SK["fta"]] = raw["fga"], raw["fta"]
        mu[..., SK["fgx"]] = raw["fga"] * (np.clip(fgp, 0.3, 0.75) - LG_FG)
        mu[..., SK["ftx"]] = raw["fta"] * (np.clip(ftp, 0.3, 0.98) - LG_FT)
        for k in ("tpm", "reb", "ast", "stl", "blk", "tov", "pts"):
            mu[..., SK[k]] = raw[k]
        if ERROR_DRAWS == "boot":
            # GP residuals from players with a similar predicted share (heteroskedastic injury risk)
            edges = np.quantile(err["gp_pred"], [1 / 3, 2 / 3])
            bin_res = [err["gp_resid"][np.digitize(err["gp_pred"], edges) == b] for b in range(3)]
            pb = np.digitize(self.avail, edges)
            res = np.empty((R, P))
            for b in range(3):
                cols = np.where(pb == b)[0]
                res[:, cols] = rng.choice(bin_res[b], (R, len(cols)))
            avail = np.clip(self.avail + res, 0, 1)
        else:
            avail = np.clip(self.avail + rng.choice(err["gp_resid"], (R, P)) * s_v * em, 0, 1)
        if self.bel:
            # each simulated season draws its own market weights (one shared shock, belief.py V2)
            import belief as Bf

            b = self.bel
            beta, gamma = Bf.market_draws(b["hy"], R, rng)                      # [K, R]
            ratio = Bf.mult_from_shift(beta.T @ b["Dm"], b["A"]) / np.maximum(b["mbar"], 1e-6)
            w = em**2                                  # prior weight K / (K + n): market info fades with data
            mu = mu * (1 + (ratio - 1) * w)[..., None]
            avail = np.clip(avail + (gamma.T @ b["Dm"] - b["gshift"]) * w, 0, 1)
        return {"mu": mu, "avail": avail, "value": value_from(mu, avail, self.vscale)}


def static_value(df: pd.DataFrame, weights=None, pool_by: str = "mkt"):
    """Context-free player value: H2H-aware z-scores per category (G-score style), summed.

    Each category's per-game contribution (x availability) is scaled by sqrt(between-player var +
    weekly within-player var) so noisy categories (STL, TO, FT%) count for less (Rosenof 2024).
    Returns (total, scale); scale lets value_from() score other stat lines identically. Per-category
    z's are left in static_value.last_by_cat.
    """
    mu = df[STATK].to_numpy(float) * df["avail"].to_numpy()[:, None]
    var = per_game_var(df[STATK].to_numpy(float))
    if pool_by == "mkt":
        pool, iters = np.argsort(df["mkt"].to_numpy())[: N_TEAMS * ROSTER], 1
    else:
        pool, iters = np.argsort(-mu[:, SK["pts"]])[: N_TEAMS * ROSTER], 3
    for _ in range(iters):
        total, out, scale = np.zeros(len(df)), {}, {}
        for c, s in zip(CATS, ["fgx", "ftx", "tpm", "reb", "ast", "stl", "blk", "tov", "pts"]):
            m, v = mu[:, SK[s]], var[:, SK[s]]
            sd = np.sqrt(m[pool].var() + v[pool].mean() / 3.5)
            sign = -1.0 if c == "TO" else 1.0
            scale[c] = (SK[s], m[pool].mean(), sd, sign)
            out[c] = sign * (m - m[pool].mean()) / sd
            total += out[c] * (1.0 if weights is None else weights[CATS.index(c)])
        pool = np.argsort(-total)[: N_TEAMS * ROSTER]
    static_value.last_by_cat = out
    return total, scale


def value_from(mu: np.ndarray, avail: np.ndarray, scale: dict) -> np.ndarray:
    """static_value for arbitrary per-game lines mu [..., 11] and availability [...]."""
    tot = np.zeros(mu.shape[:-1])
    for c, (j, m0, sd, sign) in scale.items():
        tot += sign * (mu[..., j] * avail - m0) / sd
    return tot


def projection_var(df: pd.DataFrame, err: dict) -> np.ndarray:
    """Per-game variance of the projection error for each STATK, [p, stat] (marginal, from err model)."""
    out = np.zeros((len(df), len(STATK)))
    for k in Pj.ERR_STATS:
        if k in SK:
            out[:, SK[k]] = err["scale"][k] ** 2 * df[k].clip(lower=0.05).to_numpy()
    for x, a, made, sd in (("fgx", "fga", "fgm", err["fg_sd"]), ("ftx", "fta", "ftm", err["ft_sd"])):
        att = df[a].clip(lower=0.05).to_numpy()
        pct = (df[made] / df[a].clip(lower=0.05)).to_numpy()
        lg = LG_FG if x == "fgx" else LG_FT
        out[:, SK[x]] = (att * sd) ** 2 + (pct - lg) ** 2 * err["scale"][a] ** 2 * att
    return out
