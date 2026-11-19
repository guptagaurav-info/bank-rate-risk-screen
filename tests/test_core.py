import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from brs import features as F, evaluate as X

IDX = pd.bdate_range("2021-01-04", periods=600)


def test_rate_beta_recovers_known_loading():
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0, 0.01, len(IDX)), IDX).cumsum().pipe(np.exp) * 100
    y = pd.Series(rng.normal(0, 0.03, len(IDX)), IDX).cumsum() + 2
    wk_m = mkt.resample("W-FRI").last().pct_change(); wk_y = y.resample("W-FRI").last().diff()
    r = 0.8 * wk_m - 0.05 * wk_y            # weekly bank return built from the two factors
    px = (1 + r.fillna(0)).cumprod().reindex(IDX, method="ffill").to_frame("BANK")
    b = F.rate_betas(px, mkt, y, "2021-01-01", "2023-12-31", 50)
    assert abs(b.loc["BANK", "mkt_beta"] - 0.8) < 0.05 and abs(b.loc["BANK", "rate_beta"] + 0.05) < 0.01


def test_window_return_uses_close_on_or_before_dates():
    px = pd.DataFrame({"A": [100.0, 110.0, 90.0, 99.0]}, index=pd.to_datetime(["2023-03-07", "2023-03-08", "2023-03-09", "2023-05-04"]))
    assert abs(F.window_return(px, "2023-03-08", "2023-05-04")["A"] - (99 / 110 - 1)) < 1e-12


def test_balance_sheet_never_fills_missing_securities_with_zero():
    rows = [("A", "total_assets", 100), ("A", "total_deposits", 80), ("A", "stockholders_equity", 10),
            ("B", "total_assets", 100), ("B", "total_deposits", 80), ("B", "stockholders_equity", 10), ("B", "securities_and_investments", 25)]
    s = pd.DataFrame([dict(symbol=a, item_name=b, item_value=float(c), report_date="2022-12-31", period_type="annual") for a, b, c in rows])
    f = F.balance_sheet(s, "2022-12-31", "2020-12-31")
    assert np.isnan(f.loc["A", "securities_share"]) and abs(f.loc["B", "securities_share"] - 0.25) < 1e-12


def test_score_direction_and_quintile_order():
    f = pd.DataFrame({"x": np.arange(10.0), "y": -np.arange(10.0)})
    assert (F.score(f, {"x": 1}).diff().dropna() > 0).all()
    q = X.quintiles(f.x, -f.x)
    assert q["mean"].iloc[0] > q["mean"].iloc[-1]
