"""
Trading Calendar Module - Database-backed with Market Hours

Refactored from parquet file-based to PostgreSQL database-backed calendar.
Maintains 100% backwards compatibility with the original API while adding:
- Accurate market open/close times
- Early close detection
- Intelligent caching
- Support for exceptional closures

Migration from old version:
    OLD: TradingCalendar(calendar_path=CALENDAR_PATH)
    NEW: TradingCalendar()  # No path needed, loads from DB

All existing methods work identically.
"""

import asyncio
import asyncpg
import bisect
from datetime import datetime, time, timedelta
from typing import List, Optional, Tuple
from strategy_lab.config import DB_CONFIG
from strategy_lab.utils.logger import get_logger

logger = get_logger(__name__)


class TradingCalendar:
    """
    Manages trading days with database backend and intelligent caching.

    Includes market open/close times and early close detection for
    accurate intraday trading simulation.
    """

    # Class-level cache shared across all instances
    _cache: Optional[dict] = None
    _cache_timestamp: Optional[datetime] = None
    _cache_ttl: timedelta = timedelta(hours=24)

    def __init__(self, auto_load: bool = True):
        """
        Initialize the trading calendar.

        Args:
            auto_load: If True, loads calendar from DB on first access.
        """
        self._auto_load = auto_load
        self.trading_days: List[str] = []
        self._market_hours: dict = {}  # date -> (open_time, close_time, is_early_close)

        if auto_load:
            self._ensure_loaded()

    def _ensure_loaded(self):
        """Ensure calendar is loaded, using cache if available."""
        if self.trading_days:
            return

        if self._is_cache_valid():
            self.trading_days = TradingCalendar._cache['trading_days'].copy()
            self._market_hours = TradingCalendar._cache['market_hours'].copy()
            logger.debug("Loaded trading calendar from cache")
        else:
            self.load()

    @classmethod
    def _is_cache_valid(cls) -> bool:
        """Check if the class-level cache is still valid."""
        if cls._cache is None or cls._cache_timestamp is None:
            return False

        age = datetime.now() - cls._cache_timestamp
        return age < cls._cache_ttl

    def load(self):
        """Load trading calendar from database and update cache."""
        self.trading_days, self._market_hours = asyncio.run(self._load_from_db())

        # Update class-level cache
        TradingCalendar._cache = {
            'trading_days': self.trading_days.copy(),
            'market_hours': self._market_hours.copy()
        }
        TradingCalendar._cache_timestamp = datetime.now()

        logger.info(f"Loaded {len(self.trading_days)} trading days from database")

    async def _load_from_db(self) -> Tuple[List[str], dict]:
        """Fetch all trading days and market hours from PostgreSQL."""
        query = """
        SELECT date, market_open, market_close, is_early_close
        FROM trading_calendar
        WHERE is_trading_day = TRUE
        ORDER BY date
        """

        conn = await asyncpg.connect(**DB_CONFIG)
        try:
            rows = await conn.fetch(query)

            trading_days = []
            market_hours = {}

            for row in rows:
                date_str = row['date'].strftime("%Y-%m-%d")
                trading_days.append(date_str)
                market_hours[date_str] = (
                    row['market_open'],
                    row['market_close'],
                    row['is_early_close']
                )

            return trading_days, market_hours

        finally:
            await conn.close()

    def refresh(self):
        """Force refresh of the calendar from database, bypassing cache."""
        self.load()

    @classmethod
    def clear_cache(cls):
        """Clear the class-level cache. Next load will fetch from DB."""
        cls._cache = None
        cls._cache_timestamp = None

    def date_range(self, start_date: str, end_date: str) -> List[str]:
        """
        Returns a list of trading days between start_date and end_date (inclusive).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of trading days as strings in YYYY-MM-DD format
        """
        self._ensure_loaded()

        start_idx = bisect.bisect_left(self.trading_days, start_date)
        if start_idx >= len(self.trading_days):
            return []

        end_idx = bisect.bisect_right(self.trading_days, end_date) - 1
        if end_idx < 0:
            return []

        return self.trading_days[start_idx : end_idx + 1]

    def next(self, date: str, steps: int = 1) -> str:
        """
        Returns the trading date 'steps' days after the given date.

        Args:
            date: Date in YYYY-MM-DD format
            steps: Number of trading days to move forward

        Returns:
            Trading date in YYYY-MM-DD format

        Raises:
            IndexError: If the resulting date is out of bounds
        """
        self._ensure_loaded()

        idx = bisect.bisect_left(self.trading_days, date)
        new_idx = idx + steps

        if new_idx < 0 or new_idx >= len(self.trading_days):
            raise IndexError(f"Date navigation out of bounds: {date} + {steps} steps")

        return self.trading_days[new_idx]

    def previous(self, date: str, steps: int = 1) -> str:
        """
        Returns the trading date 'steps' days before the given date.

        Args:
            date: Date in YYYY-MM-DD format
            steps: Number of trading days to move backward

        Returns:
            Trading date in YYYY-MM-DD format

        Raises:
            IndexError: If the resulting date is out of bounds
        """
        self._ensure_loaded()

        idx = bisect.bisect_left(self.trading_days, date)
        new_idx = idx - steps

        if new_idx < 0 or new_idx >= len(self.trading_days):
            raise IndexError(f"Date navigation out of bounds: {date} - {steps} steps")

        return self.trading_days[new_idx]

    def is_trading_day(self, date: str) -> bool:
        """
        Check if a given date is a trading day.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            True if the date is a trading day, False otherwise
        """
        self._ensure_loaded()

        idx = bisect.bisect_left(self.trading_days, date)
        return idx < len(self.trading_days) and self.trading_days[idx] == date

    def get_market_hours(self, date: str) -> Optional[Tuple[time, time]]:
        """
        Get market open and close times for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Tuple of (open_time, close_time) or None if not a trading day
        """
        self._ensure_loaded()

        if date not in self._market_hours:
            return None

        open_time, close_time, _ = self._market_hours[date]
        return (open_time, close_time)

    def is_early_close(self, date: str) -> bool:
        """
        Check if a given trading day has an early close.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            True if the date has an early close, False otherwise
        """
        self._ensure_loaded()

        if date not in self._market_hours:
            return False

        _, _, is_early = self._market_hours[date]
        return is_early

    def get_market_close_time(self, date: str) -> Optional[time]:
        """
        Get market close time for a specific date.

        Useful for intraday simulation to know when trading stops.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Close time or None if not a trading day
        """
        hours = self.get_market_hours(date)
        return hours[1] if hours else None

    def get_market_open_time(self, date: str) -> Optional[time]:
        """
        Get market open time for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Open time or None if not a trading day
        """
        hours = self.get_market_hours(date)
        return hours[0] if hours else None

    def is_market_open_at(self, date: str, time_str: str) -> bool:
        """
        Check if the market is open at a specific time on a given date.

        Args:
            date: Date in YYYY-MM-DD format
            time_str: Time in HH:MM:SS format

        Returns:
            True if market is open at that time, False otherwise
        """
        hours = self.get_market_hours(date)
        if not hours:
            return False

        open_time, close_time = hours
        check_time = datetime.strptime(time_str, "%H:%M:%S").time()

        return open_time <= check_time <= close_time

    def current_business_date(self, hour: int = 0) -> str:
        """
        Returns today's business date based on current time.

        If before specified hour and today is a business date, returns previous business date.

        Args:
            hour: Cutoff hour (0-23). Before this hour, returns previous trading day.

        Returns:
            Trading date in YYYY-MM-DD format

        Raises:
            ValueError: If no valid business date found
        """
        self._ensure_loaded()

        today = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().hour

        idx = bisect.bisect_left(self.trading_days, today)

        if idx < len(self.trading_days) and self.trading_days[idx] == today:
            if current_hour < hour:
                if idx > 0:
                    return self.trading_days[idx - 1]
                else:
                    raise ValueError("No previous business date available")
            else:
                return today

        if idx > 0:
            return self.trading_days[idx - 1]

        raise ValueError("No valid business date found")

    def count_trading_days(self, start_date: str, end_date: str) -> int:
        """
        Count the number of trading days between two dates (inclusive).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Number of trading days
        """
        return len(self.date_range(start_date, end_date))

    def get_trading_minutes(self, date: str) -> int:
        """
        Get the number of trading minutes for a specific date.

        Useful for normalizing volume or calculating intraday metrics.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Number of trading minutes (390 for regular day, 210 for early close)
            Returns 0 if not a trading day
        """
        hours = self.get_market_hours(date)
        if not hours:
            return 0

        open_time, close_time = hours

        # Convert to datetime for subtraction
        open_dt = datetime.combine(datetime.today(), open_time)
        close_dt = datetime.combine(datetime.today(), close_time)

        delta = close_dt - open_dt
        return int(delta.total_seconds() / 60)


def adjust_dt(
    calendar: TradingCalendar,
    dt: str,
    is_intraday: bool = False,
    is_start: bool = False,
) -> str:
    """
    Normalize a date or datetime string to a valid trading day with correct times.

    For intraday data, uses actual market open/close times from the calendar,
    respecting early closes.

    Args:
        calendar: TradingCalendar instance
        dt: Date or datetime string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        is_intraday: If True, ensures time component is present
        is_start: If True, uses market open time; else uses market close time

    Returns:
        Adjusted date or datetime string
    """
    parts = dt.split()
    date = parts[0]

    # Adjust to valid trading day if needed
    if not calendar.is_trading_day(date):
        if is_start:
            date = calendar.next(date)
        else:
            date = calendar.previous(date)

    # Add time component for intraday data
    if is_intraday:
        if len(parts) > 1:
            # Time already specified
            return f"{date} {parts[1]}"
        else:
            # Use actual market hours
            if is_start:
                market_time = calendar.get_market_open_time(date)
                default_time = "09:30:00"
            else:
                market_time = calendar.get_market_close_time(date)
                default_time = "16:00:00"

            time_str = market_time.strftime("%H:%M:%S") if market_time else default_time
            return f"{date} {time_str}"

    return date


# Convenience function for getting a singleton instance
_default_calendar: Optional[TradingCalendar] = None

def get_default_calendar() -> TradingCalendar:
    """
    Get a singleton TradingCalendar instance.

    Returns:
        Shared TradingCalendar instance
    """
    global _default_calendar
    if _default_calendar is None:
        _default_calendar = TradingCalendar()
    return _default_calendar
