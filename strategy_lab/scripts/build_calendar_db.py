"""
Build Trading Calendar Script - Database Version with Market Hours

Populates the trading_calendar table in PostgreSQL with NYSE trading days
including accurate market open/close times and early close information.

Uses pandas_market_calendars for comprehensive market hours data.

Usage:
    python build_calendar.py --start-year 1980 --end-year 2030
    python build_calendar.py --update-only  # Only add missing dates
"""

import argparse
import asyncio
import asyncpg
from datetime import datetime
import pandas_market_calendars as mcal
import polars as pl
from strategy_lab.config import DB_CONFIG
from strategy_lab.utils.logger import get_logger

logger = get_logger(__name__)


async def create_calendar_table(conn: asyncpg.Connection):
    """Create the trading_calendar table if it doesn't exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS trading_calendar (
        date DATE PRIMARY KEY,
        is_trading_day BOOLEAN NOT NULL DEFAULT TRUE,
        exchange VARCHAR(10) DEFAULT 'NYSE',
        market_open TIME,
        market_close TIME,
        is_early_close BOOLEAN DEFAULT FALSE,
        holiday_name TEXT,
        notes TEXT,
        source VARCHAR(50) DEFAULT 'pandas_market_calendars',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_trading_calendar_date
    ON trading_calendar(date) WHERE is_trading_day = TRUE;

    CREATE INDEX IF NOT EXISTS idx_trading_calendar_early_close
    ON trading_calendar(date) WHERE is_early_close = TRUE;

    -- Table for manual overrides and exceptional closures
    CREATE TABLE IF NOT EXISTS calendar_overrides (
        date DATE PRIMARY KEY,
        is_trading_day BOOLEAN NOT NULL,
        market_open TIME,
        market_close TIME,
        reason TEXT NOT NULL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    await conn.execute(create_table_query)
    logger.info("Trading calendar tables created/verified")


async def get_existing_dates(conn: asyncpg.Connection) -> set:
    """Get all dates already in the calendar."""
    query = "SELECT date FROM trading_calendar"
    rows = await conn.fetch(query)
    return {row['date'] for row in rows}


def generate_calendar_data(start_year: int, end_year: int) -> pl.DataFrame:
    """
    Generate comprehensive calendar data including market hours.

    Args:
        start_year: First year to include
        end_year: Last year to include

    Returns:
        Polars DataFrame with columns: date, is_trading_day, market_open, market_close, is_early_close
    """
    # Get NYSE calendar from pandas_market_calendars
    nyse = mcal.get_calendar('NYSE')

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    # Get trading schedule (only includes trading days) - returns pandas DataFrame
    schedule_pd = nyse.schedule(start_date=start_date, end_date=end_date)

    logger.info(f"Generated {len(schedule_pd)} trading days from {start_year} to {end_year}")

    # Convert pandas DataFrame to Polars
    # Extract date from index and convert times to Eastern
    schedule_data = {
        'date': [d.date() for d in schedule_pd.index],
        'market_open': [t.tz_convert('America/New_York').time() for t in schedule_pd['market_open']],
        'market_close': [t.tz_convert('America/New_York').time() for t in schedule_pd['market_close']],
    }

    schedule_df = pl.DataFrame(schedule_data)

    # Identify early closes (normal close is 4:00 PM / 16:00)
    # Early closes are typically at 1:00 PM / 13:00
    # Since we have Python time objects (not datetime), we need to convert to comparable format
    # We'll create a helper column with close time as minutes from midnight

    # Convert Python time objects to minutes for comparison
    close_minutes = []
    for t in schedule_df['market_close']:
        close_minutes.append(t.hour * 60 + t.minute)

    schedule_df = schedule_df.with_columns([
        pl.Series("close_minutes", close_minutes, dtype=pl.Int32)
    ]).with_columns([
        (pl.col("close_minutes") < (16 * 60)).alias("is_early_close"),
        pl.lit(True).alias("is_trading_day")
    ]).drop("close_minutes")

    # Get holidays - we'll just mark non-trading days as holidays
    # pandas_market_calendars doesn't provide easy access to holiday names
    # so we'll infer them from the trading schedule
    # The holiday_name will be filled in later based on is_trading_day


    # Create full date range including non-trading days
    all_dates = pl.date_range(
        start=pl.datetime(start_year, 1, 1),
        end=pl.datetime(end_year, 12, 31),
        interval="1d",
        eager=True
    ).cast(pl.Date)

    full_df = pl.DataFrame({'date': all_dates})

    # Merge with trading days
    full_df = full_df.join(schedule_df, on='date', how='left')

    # Fill non-trading days
    full_df = full_df.with_columns([
        pl.col('is_trading_day').fill_null(False),
        pl.col('is_early_close').fill_null(False)
    ])

    # Add holiday names based on weekday
    full_df = full_df.with_columns(
        pl.when(pl.col('date').dt.weekday().is_in([6, 7]))
        .then(pl.lit('Weekend'))
        .when(~pl.col('is_trading_day'))
        .then(pl.lit('Holiday'))  # Generic holiday name for non-trading weekdays
        .otherwise(None)
        .alias('holiday_name')
    )

    return full_df


async def generate_and_insert_trading_days(
    start_year: int,
    end_year: int,
    update_only: bool = False
):
    """
    Generate trading days and insert them into the database.

    Args:
        start_year: First year to include
        end_year: Last year to include
        update_only: If True, only insert dates not already in DB
    """
    # Generate calendar data
    df = generate_calendar_data(start_year, end_year)

    logger.info(f"Generated {len(df)} total dates from {start_year} to {end_year}")
    logger.info(f"Trading days: {df['is_trading_day'].sum()}")
    logger.info(f"Early closes: {df['is_early_close'].sum()}")

    # Connect to database and insert
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        # Create table if needed
        await create_calendar_table(conn)

        # Add known exceptional closures to overrides table
        await add_known_exceptional_closures(conn)

        # Get existing dates if update_only
        existing_dates = set()
        if update_only:
            existing_dates = await get_existing_dates(conn)
            logger.info(f"Found {len(existing_dates)} existing dates in database")

        # Filter out existing dates
        if update_only:
            df = df[~df['date'].isin(existing_dates)]
            logger.info(f"Will insert {len(df)} new dates")

        if df.is_empty():
            logger.info("No new dates to insert")
            # Still apply overrides even if no new dates
            await apply_overrides(conn)
            return

        # Prepare insert query
        insert_query = """
        INSERT INTO trading_calendar (
            date, is_trading_day, exchange, market_open, market_close,
            is_early_close, holiday_name
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (date) DO UPDATE SET
            is_trading_day = EXCLUDED.is_trading_day,
            market_open = EXCLUDED.market_open,
            market_close = EXCLUDED.market_close,
            is_early_close = EXCLUDED.is_early_close,
            holiday_name = EXCLUDED.holiday_name
        """

        # Prepare values (convert None properly)
        values = []
        for row in df.iter_rows(named=True):
            values.append((
                row['date'],
                bool(row['is_trading_day']),
                'NYSE',
                row['market_open'],
                row['market_close'],
                bool(row['is_early_close']),
                row['holiday_name']
            ))

        await conn.executemany(insert_query, values)
        logger.info(f"Successfully inserted/updated {len(values)} dates")

        # Apply any manual overrides
        await apply_overrides(conn)

        # Show summary statistics
        stats_query = """
        SELECT
            COUNT(*) as total_days,
            COUNT(*) FILTER (WHERE is_trading_day = TRUE) as trading_days,
            COUNT(*) FILTER (WHERE is_trading_day = FALSE) as non_trading_days,
            COUNT(*) FILTER (WHERE is_early_close = TRUE) as early_closes,
            COUNT(*) FILTER (WHERE source = 'manual_override') as manual_overrides,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM trading_calendar
        """
        stats = await conn.fetchrow(stats_query)

        logger.info(f"""
Calendar Statistics:
  Total dates: {stats['total_days']}
  Trading days: {stats['trading_days']}
  Non-trading days: {stats['non_trading_days']}
  Early closes: {stats['early_closes']}
  Manual overrides: {stats['manual_overrides']}
  Date range: {stats['earliest_date']} to {stats['latest_date']}
        """)

    finally:
        await conn.close()


async def apply_overrides(conn: asyncpg.Connection):
    """
    Apply manual overrides from calendar_overrides table to trading_calendar.

    This allows for corrections and exceptional closures not in pandas_market_calendars.
    """
    override_query = """
    SELECT date, is_trading_day, market_open, market_close, reason
    FROM calendar_overrides
    ORDER BY date
    """

    overrides = await conn.fetch(override_query)

    if not overrides:
        logger.info("No overrides to apply")
        return

    logger.info(f"Applying {len(overrides)} calendar overrides")

    update_query = """
    INSERT INTO trading_calendar (
        date, is_trading_day, exchange, market_open, market_close,
        is_early_close, holiday_name, source
    )
    VALUES ($1, $2, 'NYSE', $3, $4, FALSE, $5, 'manual_override')
    ON CONFLICT (date) DO UPDATE SET
        is_trading_day = EXCLUDED.is_trading_day,
        market_open = EXCLUDED.market_open,
        market_close = EXCLUDED.market_close,
        holiday_name = EXCLUDED.holiday_name,
        source = EXCLUDED.source,
        updated_at = CURRENT_TIMESTAMP
    """

    for override in overrides:
        await conn.execute(
            update_query,
            override['date'],
            override['is_trading_day'],
            override['market_open'],
            override['market_close'],
            override['reason']
        )
        logger.info(f"Applied override: {override['date']} - {override['reason']}")


async def add_known_exceptional_closures(conn: asyncpg.Connection):
    """
    Add known exceptional closures that might not be in pandas_market_calendars.

    These are historical events that caused unexpected market closures.
    """
    exceptional_closures = [
        # 9/11 terrorist attacks - confirmed in pandas_market_calendars
        # These should already be handled, but included for reference

        # Hurricane Sandy - verify these are included
        ("2012-10-29", "Hurricane Sandy"),
        ("2012-10-30", "Hurricane Sandy"),

        # President Ford funeral
        ("2007-01-02", "President Ford Funeral"),

        # President Reagan funeral
        ("2004-06-11", "President Reagan Funeral"),

        # President Nixon funeral
        ("1994-04-27", "President Nixon Funeral"),
    ]

    insert_query = """
    INSERT INTO calendar_overrides (date, is_trading_day, reason, notes)
    VALUES ($1, FALSE, $2, 'Exceptional closure - verify in pandas_market_calendars')
    ON CONFLICT (date) DO NOTHING
    """

    count = 0
    for date_str, reason in exceptional_closures:
        try:
            result = await conn.execute(
                insert_query,
                datetime.strptime(date_str, "%Y-%m-%d").date(),
                reason
            )
            if result == "INSERT 0 1":
                count += 1
        except Exception as e:
            logger.warning(f"Could not insert override for {date_str}: {e}")

    if count > 0:
        logger.info(f"Added {count} exceptional closure overrides")
    else:
        logger.info("All exceptional closures already present")


async def add_custom_closure(date: str, reason: str, notes: str = None):
    """
    Manually add a market closure or override (e.g., emergency closure, correction).

    Args:
        date: Date in YYYY-MM-DD format
        reason: Reason for closure/override
        notes: Optional additional notes
    """
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        await create_calendar_table(conn)

        # Add to overrides table
        query = """
        INSERT INTO calendar_overrides (date, is_trading_day, reason, notes)
        VALUES ($1, FALSE, $2, $3)
        ON CONFLICT (date) DO UPDATE SET
            is_trading_day = FALSE,
            reason = EXCLUDED.reason,
            notes = EXCLUDED.notes
        """

        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        await conn.execute(query, date_obj, reason, notes)

        # Apply override to main calendar
        await apply_overrides(conn)

        logger.info(f"Added/updated override for {date}: {reason}")

    finally:
        await conn.close()


async def list_overrides():
    """List all manual overrides in the calendar."""
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        query = """
        SELECT date, is_trading_day, reason, notes, created_at
        FROM calendar_overrides
        ORDER BY date DESC
        """
        rows = await conn.fetch(query)

        if not rows:
            print("No manual overrides found")
            return

        print(f"\nManual Calendar Overrides ({len(rows)} total):")
        print("-" * 80)
        print(f"{'Date':<12} {'Trading?':<10} {'Reason':<30} {'Created':<20}")
        print("-" * 80)

        for row in rows:
            trading = "YES" if row['is_trading_day'] else "NO"
            created = row['created_at'].strftime("%Y-%m-%d %H:%M")
            print(f"{row['date']!s:<12} {trading:<10} {row['reason']:<30} {created:<20}")

    finally:
        await conn.close()

async def verify_calendar():
    """Print some sample dates from the calendar for verification."""
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        # Get some recent trading days including early closes
        query = """
        SELECT date, is_trading_day, market_open, market_close,
               is_early_close, holiday_name
        FROM trading_calendar
        WHERE date >= CURRENT_DATE - INTERVAL '60 days'
        ORDER BY date
        LIMIT 30
        """
        rows = await conn.fetch(query)

        print("\nRecent calendar entries:")
        print("-" * 80)
        print(f"{'Date':<12} {'Status':<10} {'Open':<10} {'Close':<10} {'Notes':<30}")
        print("-" * 80)

        for row in rows:
            if row['is_trading_day']:
                status = "TRADING"
                open_time = row['market_open'].strftime("%H:%M") if row['market_open'] else "N/A"
                close_time = row['market_close'].strftime("%H:%M") if row['market_close'] else "N/A"
                notes = "EARLY CLOSE" if row['is_early_close'] else ""
            else:
                status = "CLOSED"
                open_time = "-"
                close_time = "-"
                notes = row['holiday_name'] or ""

            print(f"{row['date']!s:<12} {status:<10} {open_time:<10} {close_time:<10} {notes:<30}")

        # Show early close summary
        early_close_query = """
        SELECT date, market_open, market_close
        FROM trading_calendar
        WHERE is_early_close = TRUE
        AND date >= CURRENT_DATE - INTERVAL '365 days'
        ORDER BY date DESC
        LIMIT 10
        """
        early_rows = await conn.fetch(early_close_query)

        if early_rows:
            print("\n\nRecent early closes:")
            print("-" * 50)
            for row in early_rows:
                print(f"{row['date']}: {row['market_close'].strftime('%H:%M')} close")

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build trading calendar in PostgreSQL database with market hours"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1980,
        help="Start year (inclusive)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2030,
        help="End year (inclusive)"
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Only insert dates not already in database"
    )
    parser.add_argument(
        "--add-closure",
        nargs="+",
        metavar=("DATE", "REASON"),
        help="Add a custom market closure: DATE 'REASON' (reason can have spaces)"
    )
    parser.add_argument(
        "--list-overrides",
        action="store_true",
        help="List all manual calendar overrides"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify calendar by showing recent entries"
    )

    args = parser.parse_args()

    if args.add_closure:
        if len(args.add_closure) < 2:
            print("Error: --add-closure requires DATE and REASON")
            return
        date = args.add_closure[0]
        reason = " ".join(args.add_closure[1:])  # Join all remaining args as reason
        asyncio.run(add_custom_closure(date, reason))
    elif args.list_overrides:
        asyncio.run(list_overrides())
    elif args.verify:
        asyncio.run(verify_calendar())
    else:
        asyncio.run(generate_and_insert_trading_days(
            args.start_year,
            args.end_year,
            args.update_only
        ))

        # Verify after building
        if not args.update_only:
            asyncio.run(verify_calendar())


if __name__ == "__main__":
    main()
