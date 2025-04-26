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
            date_col (str): Name of the date column to align splits. Default is 'date'.

        Returns:
            pl.DataFrame: Adjusted DataFrame.
        """
        if splits.is_empty():
            return df

        df = df.sort(date_col)
        splits = splits.sort("split_date")

        split_dates = splits.get_column("split_date").to_list()
        split_ratios = splits.get_column("split_ratio").to_list()

        cumulative_factor = 1.0
        factors = []
        idx = 0

        for date in df.get_column(date_col).to_list():
            while idx < len(split_dates) and date >= split_dates[idx]:
                cumulative_factor *= split_ratios[idx]
                idx += 1
            factors.append(cumulative_factor)

        adjustment = pl.Series("adjustment", factors)

        # Apply adjustment factors vectorially
        df = df.with_columns([
            (pl.col("open") / adjustment).alias("open"),
            (pl.col("high") / adjustment).alias("high"),
            (pl.col("low") / adjustment).alias("low"),
            (pl.col("close") / adjustment).alias("close"),
            (pl.col("volume") * adjustment).alias("volume"),
        ])

        return df
