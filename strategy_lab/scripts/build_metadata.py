import polars as pl
from pathlib import Path

def build_metadata(etf_output: Path, stock_output: Path, finance_db_path: Path, as_of_date: str = None) -> None:
    df_etfs = pl.read_csv(finance_db_path / "data/etfs/etfs.csv")
    df_stocks = pl.read_csv(finance_db_path / "data/stocks/stocks.csv")

    if as_of_date:
        df_etfs = df_etfs.filter(pl.col("last_updated") <= as_of_date)
        df_stocks = df_stocks.filter(pl.col("last_updated") <= as_of_date)

    us_etfs = df_etfs.filter(pl.col("country") == "United States").select(["symbol", "name"])
    us_stocks = df_stocks.filter(pl.col("country") == "United States").select(["symbol", "name", "sector", "industry"])

    us_etfs.write_parquet(etf_output)
    us_stocks.write_parquet(stock_output)

if __name__ == "__main__":
    finance_db_repo = Path("path/to/FinanceDatabase")
    build_metadata(
        etf_output=Path("path/to/etfs.parquet"),
        stock_output=Path("path/to/stocks.parquet"),
        finance_db_path=finance_db_repo
    )
