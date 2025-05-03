# trading_bot/strategies/base.py

from abc import ABC, abstractmethod
from trading_bot.session import TradingSession

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, session: TradingSession):
        self.session = session

    @abstractmethod
    def run(self, **kwargs):
        """Run the strategy logic."""
        pass
