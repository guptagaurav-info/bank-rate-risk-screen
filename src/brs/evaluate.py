"""Rank tests: does a higher risk score line up with a worse return?"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ic(x, y):
    d = pd.concat([x, y], axis=1).dropna()
    r, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return {"spearman": float(r), "p": float(p), "n": int(len(d))}


def quintiles(x, y):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    d["q"] = pd.qcut(d.x.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    g = d.groupby("q", observed=True).y.agg(["mean", "median", "count"])
    return g


def boot_spread(x, y, n=2000, seed=7):
    """Bootstrap CI for (riskiest fifth mean return - safest fifth mean return)."""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        q = pd.qcut(b.x.rank(method="first"), 5, labels=False)
        out.append(b.y[q == 4].mean() - b.y[q == 0].mean())
    o = np.array(out)
    return {"lo": float(np.quantile(o, 0.025)), "hi": float(np.quantile(o, 0.975))}
