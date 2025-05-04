import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import holidays

def generate_trading_calendar(start_year: int, end_year: int, output_path: Path) -> None:
    us_holidays = holidays.NYSE(years=range(start_year, end_year + 1))

    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    trading_days = []
    while current_date <= end_date:
        if current_date.weekday() < 5 and current_date.date() not in us_holidays:
            trading_days.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    pl.DataFrame({"date": trading_days}).write_parquet(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build trading calendar parquet file.")
    parser.add_argument("--start_year", type=int, default=1980, help="Start year (inclusive)")
    parser.add_argument("--end_year", type=int, default=2030, help="End year (inclusive)")
    parser.add_argument("--output", type=str, required=True, help="Output path for the calendar.parquet")
    args = parser.parse_args()

    output_file = Path(args.output)
    generate_trading_calendar(start_year=args.start_year, end_year=args.end_year, output_path=output_file)
