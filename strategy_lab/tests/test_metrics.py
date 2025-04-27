import pytest
from strategy_lab.backtest.metrics import MetricsCalculator

def test_compute_pnl():
    trades = [
        {"timestamp": "2024-01-01 09:30", "side": "buy", "quantity": 100, "price": 10},
        {"timestamp": "2024-01-01 16:00", "side": "sell", "quantity": 100, "price": 11},
    ]
    pnl = MetricsCalculator.compute_pnl(trades)
    assert pnl == 100  # (11 - 10) * 100

def test_compute_daily_returns():
    trades = {
        "AAPL": [
            {"timestamp": "2024-01-01 09:30", "side": "buy", "quantity": 100, "price": 10},
            {"timestamp": "2024-01-01 16:00", "side": "sell", "quantity": 100, "price": 11},
        ]
    }
    daily_returns = MetricsCalculator.compute_daily_returns(trades)
    assert daily_returns.height == 1
    assert daily_returns[0, "cash_flow"] == 100
