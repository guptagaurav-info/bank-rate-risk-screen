"""Run the 2023 bank-stress test end to end: signals known beforehand vs returns during each window."""
import json, os, hashlib
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
from brs import data as D, features as F, evaluate as X

FIG = "reports/figures/"
NAVY, GOLD, GREY, RED, PURP = "#1F3A5F", "#B08D57", "#8A8F98", "#A23B3B", "#5B4B8A"
SIGNALS = ["securities_share", "aoci_to_equity", "rate_beta", "deposit_growth", "loans_to_deposits",
           "equity_to_assets", "mkt_beta", "log_assets"]
LABEL = {"securities_share": "Securities / assets", "aoci_to_equity": "Unrealised losses (AOCI) / equity",
         "rate_beta": "Stock beta to the 10-year yield", "deposit_growth": "Deposit growth 2020-22",
         "loans_to_deposits": "Loans / deposits", "equity_to_assets": "Equity / assets",
         "mkt_beta": "Stock beta to the S&P 500", "log_assets": "Size (log assets)",
         "score": "Textbook rate-risk score", "funding": "Funding-fragility score"}
FUNDING = {"deposit_growth": 1, "loans_to_deposits": 1, "equity_to_assets": -1}  # defined after the first run; tested on episode 2


def frame(cfg, px, spy, y, s, bw, asof, base, win):
    b = F.rate_betas(px, spy, y.bc_10year, *bw, cfg["min_weeks"])
    f = F.balance_sheet(s, asof, base).join(b, how="inner").join(F.window_return(px, *win))
    p0 = px.loc[:win[0]].iloc[-1]
    f = f[p0.reindex(f.index) >= cfg["min_price"]].copy()
    f["score"] = F.score(f, cfg["directions"]); f["funding"] = F.score(f, FUNDING)
    return f


def exclusion(f, col):
    """Equal-weight return of the whole set vs only its safer half by `col` (one simple, symmetric rule)."""
    d = f[[col, "ret"]].dropna()
    med = d[col].median()
    return {"all": float(d.ret.mean()), "safer_half": float(d[d[col] <= med].ret.mean()),
            "riskier_half": float(d[d[col] > med].ret.mean()), "n": int(len(d))}


def main():
    cfg = yaml.safe_load(open("configs/default.yaml")); os.makedirs(FIG, exist_ok=True)
    px, et, y, s = D.prices(), D.etfs(), D.yields(), D.statements()
    W = {"crisis": (cfg["beta_window"], cfg["asof_balance_sheet"], cfg["deposit_base_date"], cfg["crisis"]),
         "episode2": (cfg["episode2"]["beta_window"], cfg["episode2"]["asof_balance_sheet"], cfg["deposit_base_date"], cfg["episode2"]["window"]),
         "placebo": (["2020-06-01", "2022-03-04"], "2021-12-31", cfg["deposit_base_date"], cfg["placebo"]),
         "strict_lag": (cfg["beta_window"], cfg["strict_lag"]["asof_balance_sheet"], cfg["strict_lag"]["deposit_base_date"], cfg["crisis"])}
    fr = {k: frame(cfg, px, et.SPY, y, s, *v) for k, v in W.items()}
    spy = lambda w: float(et.SPY.loc[:w[1]].iloc[-1] / et.SPY.loc[:w[0]].iloc[-1] - 1)
    y10 = lambda w: [float(y.bc_10year.loc[:w[0]].iloc[-1]), float(y.bc_10year.loc[:w[1]].iloc[-1])]
    m = {"universe": {"banks_in_profile": int(len(D.universe())), "crisis_sample": int(len(fr["crisis"]))},
         "windows": {k: {"dates": v[3], "bank_ew_return": float(fr[k].ret.mean()), "bank_median_return": float(fr[k].ret.median()),
                         "spy_return": spy(v[3]), "ten_year": y10(v[3]), "n": int(len(fr[k]))} for k, v in W.items()},
         "ic": {k: {c: X.ic(fr[k][c], fr[k].ret) for c in SIGNALS + ["score", "funding"]} for k in fr},
         "quintiles_crisis": {c: X.quintiles(fr["crisis"][c], fr["crisis"].ret)["mean"].round(4).tolist() for c in ["score", "funding", "mkt_beta"]},
         "quintiles_episode2": {c: X.quintiles(fr["episode2"][c], fr["episode2"].ret)["mean"].round(4).tolist() for c in ["score", "funding", "mkt_beta"]},
         "spread_ci_crisis": {c: X.boot_spread(fr["crisis"][c], fr["crisis"].ret) for c in ["score", "funding", "mkt_beta"]},
         "spread_ci_episode2": {c: X.boot_spread(fr["episode2"][c], fr["episode2"].ret) for c in ["score", "funding", "mkt_beta"]},
         "exclusion": {k: {c: exclusion(fr[k], c) for c in ["score", "funding", "mkt_beta"]} for k in ["crisis", "episode2"]},
         "medians_crisis": fr["crisis"][SIGNALS].median().round(4).to_dict(),
         "worst10_crisis": fr["crisis"].sort_values("ret").head(10)[["ret", "loans_to_deposits", "deposit_growth", "securities_share", "mkt_beta"]].round(3).reset_index().to_dict("records"),
         "data_sha256": {f: hashlib.sha256(open("data/" + f, "rb").read()).hexdigest() for f in sorted(os.listdir("data")) if f.endswith(".parquet")}}
    book = float(cfg["illustrative_book_usd"])
    m["dollars_per_1bn"] = {k: {c: (v["safer_half"] - v["all"]) * book for c, v in m["exclusion"][k].items()} for k in m["exclusion"]}
    json.dump(m, open("reports/metrics.json", "w"), indent=2, default=float)
    fr["crisis"].to_csv("reports/crisis_panel.csv")
    figures(m, fr, px, et, y)
    print(json.dumps({k: {c: round(v["spearman"], 3) for c, v in d.items()} for k, d in m["ic"].items()}, indent=1))
    print(json.dumps(m["exclusion"], indent=1)); print(json.dumps(m["windows"], indent=1))


def figures(m, fr, px, et, y):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # 1 rank correlation by signal, crisis vs episode 2
    cols = SIGNALS + ["score", "funding"]
    fig, ax = plt.subplots(figsize=(8.6, 5.6)); yy = np.arange(len(cols)); h = 0.38
    for j, (k, c, lab) in enumerate([("crisis", NAVY, "Mar-May 2023 bank run"), ("episode2", GOLD, "Aug-Oct 2023 yield spike")]):
        v = [m["ic"][k][s]["spearman"] for s in cols]
        ax.barh(yy + (j - 0.5) * h, v, h, color=c, label=lab)
    ax.axvline(0, color="black", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels([LABEL[c] for c in cols]); ax.invert_yaxis()
    ax.set_xlabel("Rank correlation with the bank's return in the window\n(negative = higher signal, bigger loss)")
    ax.set_title("What lined up with the damage, signal by signal", pad=28); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG + "signals.png", dpi=160); plt.close(fig)
    # 2 quintiles, crisis
    fig, ax = plt.subplots(figsize=(8.6, 4.8)); x = np.arange(5); w = 0.27
    for j, (c, col) in enumerate([("score", GREY), ("funding", RED), ("mkt_beta", NAVY)]):
        v = m["quintiles_crisis"][c]; ax.bar(x + (j - 1) * w, v, w, color=col, label=LABEL[c])
    ax.set_xticks(x); ax.set_xticklabels(["Safest fifth", "2", "3", "4", "Riskiest fifth"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, 0)); ax.set_ylabel("Average bank stock return, 8 Mar - 4 May 2023")
    ax.set_title("Bank losses in the 2023 run, sorted by each screen"); ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG + "quintiles.png", dpi=160); plt.close(fig)
    # 3 scatter: loans/deposits vs crisis return
    f = fr["crisis"][["loans_to_deposits", "ret"]].dropna(); f = f[f.loans_to_deposits.between(0.2, 1.6)]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.scatter(f.loans_to_deposits, f.ret, s=12, color=NAVY, alpha=0.5)
    b = pd.qcut(f.loans_to_deposits, 10); g = f.groupby(b, observed=True).agg(x=("loans_to_deposits", "median"), y=("ret", "median"))
    ax.plot(g.x, g.y, color=RED, lw=2.5, marker="o", label="Median by tenth")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, 0)); ax.yaxis.set_major_formatter(PercentFormatter(1.0, 0))
    ax.set_xlabel("Loans as a share of deposits, end-2022"); ax.set_ylabel("Stock return, 8 Mar - 4 May 2023")
    ax.set_title("Lending out more of the deposit base meant bigger losses, up to a point"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG + "loans_deposits.png", dpi=160); plt.close(fig)
    # 4 timeline: equal-weight bank index vs SPY, 10-year yield
    r = px.loc["2021-01-01":"2024-06-30"].pct_change(fill_method=None).clip(-0.5, 0.5).mean(axis=1)
    idx = (1 + r.fillna(0)).cumprod(); sp = et.SPY.loc["2021-01-01":"2024-06-30"]; sp = sp / sp.iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.8)); ax.plot(idx.index, idx, color=NAVY, lw=2, label="Bank stocks, equal weight")
    ax.plot(sp.index, sp, color=GREY, lw=1.6, label="S&P 500 (SPY)")
    for (a, b2), lab in [(("2023-03-08", "2023-05-04"), "Bank run"), (("2023-08-15", "2023-10-27"), "Yield spike")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b2), color=GOLD, alpha=0.25)
        ax.text(pd.Timestamp(a), 1.45, " " + lab, fontsize=9, color="#6b5530")
    ax2 = ax.twinx(); yl = y.bc_10year.loc["2021-01-01":"2024-06-30"]; ax2.plot(yl.index, yl, color=RED, lw=1.2, ls="--", label="10-year yield (right)")
    ax2.set_ylabel("10-year Treasury yield, %"); ax.set_ylabel("Growth of $1"); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Bank stocks, the market and the 10-year yield, 2021-2024"); ax2.spines["top"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG + "timeline.png", dpi=160); plt.close(fig)
    # 5 economics
    fig, ax = plt.subplots(figsize=(8.4, 4.6)); ks = ["score", "funding", "mkt_beta"]; x = np.arange(3); w = 0.38
    for j, (k, col, lab) in enumerate([("crisis", NAVY, "Mar-May 2023"), ("episode2", GOLD, "Aug-Oct 2023 (out of sample)")]):
        v = [m["dollars_per_1bn"][k][c] / 1e6 for c in ks]
        ax.bar(x + (j - 0.5) * w, v, w, color=col, label=lab)
        for xi, vi in zip(x + (j - 0.5) * w, v): ax.text(xi, vi + (0.6 if vi >= 0 else -2.6), (f"+\\${vi:.0f}m" if vi >= 0 else f"-\\${-vi:.0f}m"), ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in ks], fontsize=10)
    ax.set_ylabel("Loss avoided, \\$m per \\$1bn\n(hold the safer half vs hold all)"); ax.set_title("What each screen would have saved an equal-weight bank book")
    ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(FIG + "economics.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
