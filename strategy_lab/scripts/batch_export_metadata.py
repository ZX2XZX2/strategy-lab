import argparse
from pathlib import Path
from datetime import datetime, timedelta
from export_metadata_by_date import export_metadata_by_date

def batch_export(finance_db_path: Path, output_base: Path, start_date: str, end_date: str) -> None:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date = start

    while date <= end:
        target_date = date.strftime("%Y-%m-%d")
        etf_file = output_base / f"etfs_{target_date}.parquet"
        stock_file = output_base / f"stocks_{target_date}.parquet"

        if etf_file.exists() and stock_file.exists():
            print(f"Skipping {target_date} (already exists)")
        else:
            print(f"Exporting metadata for {target_date}...")
            export_metadata_by_date(finance_db_path, output_base, target_date)

        date += timedelta(days=30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch export ETF and Stock metadata monthly.")
    parser.add_argument("--finance_db_path", type=str, required=True, help="Path to FinanceDatabase repo")
    parser.add_argument("--output_base", type=str, required=True, help="Directory to output parquet files")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    batch_export(
        finance_db_path=Path(args.finance_db_path),
        output_base=Path(args.output_base),
        start_date=args.start_date,
        end_date=args.end_date
    )