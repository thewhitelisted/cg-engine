"""Item 3: honest uncertainty for strategy comparisons across real seasons.

Each season gives a paired effect d_s = mean over drafts of (strategy - reference) with within-season
standard error se_s (draft-to-draft). Seasons differ (one realised NBA season each), so effects are
pooled with a DerSimonian-Laird random-effects meta-analysis; the CI widens with between-season
heterogeneity tau^2. Power: seasons needed to detect a true effect delta at alpha 0.05 / power 0.8.

    python3 analysis.py data/realtest_dev.csv [reference] [metric]
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm


def per_season(res, strat, ref, metric):
    out = []
    for t, g in res.groupby("season"):
        p = g.pivot(index="seed", columns="strategy", values=metric)
        if strat not in p or ref not in p:
            continue
        d = (p[strat] - p[ref]).dropna()
        out.append((t, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)))
    return pd.DataFrame(out, columns=["season", "d", "se", "n"])


def random_effects(ps: pd.DataFrame):
    w = 1 / ps.se.clip(lower=1e-9) ** 2
    fixed = (w * ps.d).sum() / w.sum()
    k = len(ps)
    Q = (w * (ps.d - fixed) ** 2).sum()
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (ps.se**2 + tau2)
    est = (wr * ps.d).sum() / wr.sum()
    se = np.sqrt(1 / wr.sum())
    return {"fixed": fixed, "fixed_se": np.sqrt(1 / w.sum()), "re": est, "re_se": se,
            "re_ci": (est - 1.96 * se, est + 1.96 * se), "tau": np.sqrt(tau2), "k": k}


def seasons_needed(delta, tau, se_within, alpha=0.05, power=0.8):
    """Seasons so a random-effects test detects true mean effect delta (per-season variance tau^2 + se^2)."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil((z**2) * (tau**2 + se_within**2) / delta**2))


def main(path, ref="tool", metric="champ"):
    res = pd.read_csv(path)
    print(f"{path}: metric={metric}, reference={ref}")
    for s in sorted(res.strategy.unique()):
        if s == ref:
            continue
        ps = per_season(res, s, ref, metric)
        if ps.empty:
            continue
        r = random_effects(ps)
        per = "  ".join(f"{int(t)}: {d:+.4f}±{se:.4f}" for t, d, se, _ in ps.itertuples(index=False))
        print(f"  {s:15} pooled {r['re']:+.4f} (95% CI {r['re_ci'][0]:+.4f}, {r['re_ci'][1]:+.4f}), "
              f"between-season sd {r['tau']:.4f} | {per}")
    ps = per_season(res, "adp", ref, metric)
    if not ps.empty:
        r = random_effects(ps)
        se_w = ps.se.mean()
        print("\npower (vs adp's observed season spread): seasons needed to confirm a true edge of")
        for delta in (0.01, 0.02, 0.04, 0.08):
            print(f"  {delta:.2f}: {seasons_needed(delta, r['tau'], se_w)} seasons")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "data/realtest_dev.csv", *(a[1:3]))
