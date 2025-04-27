from typing import List
import polars as pl

class MetricsCalculator:
    @staticmethod
    def compute_pnl(trades: List[dict]) -> float:
        """
        Compute total PnL (Profit and Loss) from a list of trades.
        """
        pnl = 0.0
        position = 0
        avg_cost = 0.0

        for trade in trades:
            qty = trade["quantity"]
            price = trade["price"]
            side = trade["side"]

            if side == "buy":
                new_position = position + qty
                avg_cost = (avg_cost * position + price * qty) / new_position
                position = new_position
            elif side == "sell":
                pnl += qty * (price - avg_cost)
                position -= qty

        return pnl

    @staticmethod
    def compute_daily_returns(trade_results: dict) -> pl.DataFrame:
        """
        Aggregate trades by day and compute daily PnL returns.
        """
        records = []
        for ticker, trades in trade_results.items():
            for trade in trades:
                date = trade["timestamp"].split(" ")[0]
                price = trade["price"]
                side = trade["side"]
                quantity = trade["quantity"]
                sign = 1 if side == "sell" else -1
                records.append((date, sign * price * quantity))

        df = pl.DataFrame(records, schema=["date", "cash_flow"])
        daily_pnl = df.groupby("date").sum()
        return daily_pnl