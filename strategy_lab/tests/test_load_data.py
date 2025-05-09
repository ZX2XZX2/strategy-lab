import polars as pl
from strategy_lab.data.loader import DataLoader
from strategy_lab.utils.trading_calendar import TradingCalendar

def print_data(df: pl.DataFrame, title: str):
    print(f"{title}")
    if df.is_empty():
        print("[No Data]")
    else:
        with pl.Config(tbl_rows=-1):
            print(df)
    print("\n")

def test_load_data(ticker: str, start_date: str = None, end_date: str = None):
    trading_calendar = TradingCalendar()
    loader = DataLoader(trading_calendar)

    # Load EOD data
    try:
        eod_df = loader.load_eod(ticker, start_date=start_date, end_date=end_date, as_of_date=end_date)
        print_data(eod_df, f"EOD Data for {ticker} (as of {end_date})")
        print_data(eod_df, f"EOD Data for {ticker}")
    except Exception as e:
        print(f"Error loading EOD data: {e}")

    # Load Intraday data
    try:
        intraday_df = loader.load_intraday(ticker, start_date, end_date)
        print_data(intraday_df, f"Intraday Data for {ticker} ({start_date} to {end_date})")
    except Exception as e:
        print(f"Error loading Intraday data: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and print EOD and Intraday data for a given ticker and date range.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for intraday data (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date for intraday data (YYYY-MM-DD)")
    args = parser.parse_args()
    test_load_data(args.ticker, args.start_date, args.end_date)
