import argparse
import os
from pathlib import Path
from git import Repo
from datetime import datetime
from build_metadata import build_metadata

def export_metadata_by_date(finance_db_path: Path, output_base: Path, target_date: str) -> None:
    repo = Repo(finance_db_path)
    dt = datetime.strptime(target_date, "%Y-%m-%d")

    commits = list(repo.iter_commits("main"))
    best_commit = max(
        (c for c in commits if datetime.fromtimestamp(c.committed_date) <= dt),
        key=lambda c: c.committed_date,
        default=None
    )

    if best_commit is None:
        raise ValueError(f"No commit found before {target_date}")

    repo.git.checkout(best_commit.hexsha)

    build_metadata(
        etf_output=output_base / f"etfs_{target_date}.parquet",
        stock_output=output_base / f"stocks_{target_date}.parquet",
        finance_db_path=finance_db_path,
        as_of_date=target_date
    )

    repo.git.checkout("main")  # Reset back to latest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ETF and Stock metadata as of a given date.")
    parser.add_argument("--finance_db_path", type=str, required=True, help="Path to FinanceDatabase repo")
    parser.add_argument("--output_base", type=str, required=True, help="Directory to output parquet files")
    parser.add_argument("--target_date", type=str, required=True, help="As-of date (YYYY-MM-DD)")
    args = parser.parse_args()

    export_metadata_by_date(
        finance_db_path=Path(args.finance_db_path),
        output_base=Path(args.output_base),
        target_date=args.target_date
    )