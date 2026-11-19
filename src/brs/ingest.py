"""Download the four public mirror files, check pinned sha256, and write the small extracts in data/."""
import hashlib, os, urllib.request, yaml
import duckdb

RAW = "data/raw/"
ITEMS = ("total_assets", "total_deposits", "non_interest_bearing_deposits", "held_to_maturity_securities",
         "available_for_sale_securities", "gains_losses_not_affecting_retained_earnings", "stockholders_equity",
         "tangible_book_value", "net_loan", "securities_and_investments", "interest_expense_for_deposit",
         "net_interest_income", "long_term_debt", "other_current_borrowings", "cash_and_due_from_banks",
         "goodwill_and_other_intangible_assets")


def fetch(cfg, local=None):
    os.makedirs(RAW, exist_ok=True)
    for f, s in cfg["sources"].items():
        p = RAW + f
        if not os.path.exists(p):
            if local and os.path.exists(os.path.join(local, f)):
                os.symlink(os.path.abspath(os.path.join(local, f)), p)
            else:
                urllib.request.urlretrieve(s["url"], p)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        print(f, h[:12], "OK" if h == s["sha256"] else "MISMATCH (mirror updated; results may differ)")


def extract():
    c = duckdb.connect()
    c.sql(f"create table b as select symbol, industry from '{RAW}stock_profile.parquet' where industry in ('Banks-Regional','Banks-Diversified')")
    c.sql("copy (select * from b) to 'data/universe.parquet'")
    c.sql(f"copy (select p.symbol, p.report_date, p.close, p.volume from '{RAW}stock_prices.parquet' p join b using(symbol) where p.report_date >= '2020-06-01') to 'data/bank_prices.parquet'")
    c.sql(f"copy (select symbol, report_date, close from '{RAW}stock_prices.parquet' where symbol = 'SPY' and report_date >= '2020-06-01') to 'data/etf_prices.parquet'")
    items = "(" + ",".join(f"'{i}'" for i in ITEMS) + ")"
    c.sql(f"copy (select s.* from '{RAW}stock_statement.parquet' s join b using(symbol) where item_name in {items} and period_type = 'annual' and report_date between '2019-06-01' and '2023-12-31') to 'data/bank_statements.parquet'")
    c.sql(f"copy (select * from '{RAW}daily_treasury_yield.parquet' where report_date >= '2020-06-01') to 'data/yields.parquet'")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    fetch(cfg, os.environ.get("LOCAL_MIRROR")); extract()
