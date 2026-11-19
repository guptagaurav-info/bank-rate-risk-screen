"""Load the extracted Hugging Face mirror files (prices split-adjusted, not dividend-adjusted)."""
import pandas as pd

D = "data/"


def prices():
    p = pd.read_parquet(D + "bank_prices.parquet")
    p["report_date"] = pd.to_datetime(p.report_date); p["close"] = p.close.astype(float)
    return p.pivot_table(index="report_date", columns="symbol", values="close")


def etfs():
    p = pd.read_parquet(D + "etf_prices.parquet")
    p["report_date"] = pd.to_datetime(p.report_date); p["close"] = p.close.astype(float)
    return p.pivot_table(index="report_date", columns="symbol", values="close")


def yields():
    y = pd.read_parquet(D + "yields.parquet")
    y["report_date"] = pd.to_datetime(y.report_date)
    y = y.set_index("report_date").astype(float)
    return y[["bc_2year", "bc_10year"]] * 100  # percent


def statements():
    s = pd.read_parquet(D + "bank_statements.parquet")
    s["item_value"] = s.item_value.astype(float)
    return s


def universe():
    return pd.read_parquet(D + "universe.parquet")
