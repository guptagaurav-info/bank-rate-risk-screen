"""Signals known before the stress window: weekly rate betas and balance-sheet ratios."""
import numpy as np
import pandas as pd


def weekly(df):
    return df.resample("W-FRI").last()


def rate_betas(px, spy, y10, start, end, min_weeks):
    """OLS of weekly bank return on SPY return and the weekly change in the 10-year yield (per 1 pp)."""
    w = weekly(px.loc[start:end]); r = w.pct_change()
    m = weekly(spy.loc[start:end]).pct_change(); dy = weekly(y10.loc[start:end]).diff()
    X = pd.concat([m.rename("mkt"), dy.rename("dy")], axis=1).dropna()
    out = {}
    for s in r.columns:
        d = pd.concat([r[s], X], axis=1).dropna()
        if len(d) < min_weeks:
            continue
        A = np.column_stack([np.ones(len(d)), d.mkt, d.dy])
        b, *_ = np.linalg.lstsq(A, d[s].values, rcond=None)
        out[s] = {"mkt_beta": b[1], "rate_beta": b[2], "weeks": len(d)}
    return pd.DataFrame(out).T


def _item(s, name, date, period):
    x = s[(s.item_name == name) & (s.report_date == date) & (s.period_type == period)]
    return x.groupby("symbol").item_value.last()


def balance_sheet(s, asof, base):
    q = lambda n: _item(s, n, asof, "annual")
    ta, dep, eq = q("total_assets"), q("total_deposits"), q("stockholders_equity")
    # securities: the reported total where present, else HTM + AFS when both are reported (never fill missing with 0)
    sec = q("securities_and_investments").reindex(ta.index)
    sec = sec.fillna(q("held_to_maturity_securities").reindex(ta.index) + q("available_for_sale_securities").reindex(ta.index))
    aoci = q("gains_losses_not_affecting_retained_earnings")
    nib = q("non_interest_bearing_deposits"); loans = q("net_loan")
    dep0 = _item(s, "total_deposits", base, "annual")
    f = pd.DataFrame({"total_assets": ta, "securities_share": sec / ta, "aoci_to_equity": aoci / eq,
                      "deposit_growth": dep / dep0 - 1, "nib_share": nib / dep, "loans_to_deposits": loans / dep,
                      "equity_to_assets": eq / ta})
    f["log_assets"] = np.log(f.total_assets)
    return f.replace([np.inf, -np.inf], np.nan)


def window_return(px, start, end):
    a = px.loc[:start].iloc[-1]; b = px.loc[:end].iloc[-1]
    return (b / a - 1).rename("ret")


def max_drawdown(px, start, end):
    w = px.loc[start:end]; base = px.loc[:start].iloc[-1]
    return (w.min() / base - 1).rename("mdd")


def score(f, directions):
    """Pre-registered composite: mean of direction-signed cross-sectional z-scores (winsorised at 1/99%)."""
    z = []
    for c, sgn in directions.items():
        x = f[c].clip(f[c].quantile(0.01), f[c].quantile(0.99))
        z.append(sgn * (x - x.mean()) / x.std())
    return pd.concat(z, axis=1).mean(axis=1, skipna=False).rename("risk_score")
