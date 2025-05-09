# Efficient split adjustment utilities

import polars as pl

class Adjuster:

    @staticmethod
    def apply_splits(df: pl.DataFrame, splits: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """
        Efficiently adjust a Polars DataFrame (EOD or intraday) based on stock splits.

        Args:
            df (pl.DataFrame): DataFrame containing price and volume columns.
            splits (pl.DataFrame): DataFrame containing 'split_date' and 'split_ratio' sorted ascending.
            date_col (str): Name of the date or timestamp column to align splits.

        Returns:
            pl.DataFrame: Adjusted DataFrame.
        """
        if splits.is_empty():
            return df

        # Handle intraday data with a timestamp column
        if date_col == "timestamp":
            # Extract date part from timestamp as datetime.date
            df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))

        df = df.sort(date_col)
        splits = splits.sort("date", descending=True)  # Reverse to handle earlier dates first

        split_dates = splits.get_column("date").to_list()
        split_ratios = splits.get_column("ratio").to_list()

        factors = []

        # Traverse the dates and calculate cumulative factors
        for date in df.get_column("date").to_list():
            idx = 0
            cumulative_factor = 1.0
            # Adjust the cumulative factor for dates before the split
            while idx < len(split_dates) and date <= split_dates[idx]:
                cumulative_factor *= split_ratios[idx]
                idx += 1
            factors.append(cumulative_factor)

        adjustment = pl.Series("adjustment", factors)

        # Apply adjustment factors to prices and volume, converting to integers
        df = df.with_columns([
            (pl.col("open") * adjustment).round(0).cast(pl.Int64).alias("open"),
            (pl.col("high") * adjustment).round(0).cast(pl.Int64).alias("high"),
            (pl.col("low") * adjustment).round(0).cast(pl.Int64).alias("low"),
            (pl.col("close") * adjustment).round(0).cast(pl.Int64).alias("close"),
            (pl.col("volume") / adjustment).round(0).cast(pl.Int64).alias("volume"),
        ])

        # Drop the temporary date column for intraday data
        if date_col == "timestamp" and "date" in df.columns:
            df = df.drop("date")

        return df
