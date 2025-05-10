import time
import asyncio
import os
from datetime import datetime
import polars as pl
import asyncpg
import strategy_lab.config as cfg

class PostgresConnection:
    def __init__(self, db_url):
        self.db_url = db_url
        self.connection = None

    async def connect(self):
        self.connection = await asyncpg.connect(self.db_url)
        print("Postgres connection established.")

    async def close(self):
        await self.connection.close()
        print("Postgres connection closed.")

    async def load_data(self, table_name, ticker, start_date, end_date):
        query = f"SELECT * FROM {table_name} WHERE stk = '{ticker}' AND dt BETWEEN '{start_date}' AND '{end_date}'"
        start_time = time.time()
        data = await self.connection.fetch(query)
        end_time = time.time()
        print(f'Loaded from PostgreSQL using asyncpg in {end_time - start_time:.4f} seconds')
        df = pl.DataFrame(data)
        return df


def load_from_parquet(ticker, start_date, end_date):
    start_time = time.time()
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    parquet_path = cfg.EOD_DIR / f"{ticker}.parquet"
    scan = pl.scan_parquet(parquet_path)
    df = (scan.filter((pl.col('date') >= start_date) & (pl.col('date') <= end_date)).collect())
    end_time = time.time()
    print(f'Loaded from Parquet in {end_time - start_time:.4f} seconds')
    return df

async def load_from_postgres(db_url, table_name, ticker, start_date, end_date):
    conn = await asyncpg.connect(db_url)
    start_time = time.time()
    query = f"SELECT * FROM {table_name} WHERE stk = '{ticker}' AND dt BETWEEN '{start_date}' AND '{end_date}'"
    data = await conn.fetch(query)
    end_time = time.time()
    await conn.close()
    print(f'Loaded from PostgreSQL in {end_time - start_time:.4f} seconds')
    df = pl.DataFrame(data)
    return df

def load_from_postgres_polars(db_url, table_name, ticker, start_date, end_date):
    start_time = time.time()
    query = f"SELECT * FROM {table_name} WHERE stk = '{ticker}' AND dt BETWEEN '{start_date}' AND '{end_date}'"
    df = pl.read_database_uri(query, db_url)
    end_time = time.time()
    print(f'Loaded from PostgreSQL using Polars in {end_time - start_time:.4f} seconds')
    return df

async def main():
    db_url = os.getenv("POSTGRES_CNX")
    table_name = 'eods'
    ticker = 'VKTX'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    end_date_1 = '2024-01-31'

    print("Loading EOD data from Parquet file...")
    load_from_parquet(ticker, start_date, end_date)
    load_from_parquet(ticker, start_date, end_date_1)
    # print(df_parquet.head())

    pg_conn = PostgresConnection(db_url)
    await pg_conn.connect()
    print("\nLoading EOD data from PostgreSQL database using asyncpg...")
    await pg_conn.load_data(table_name, ticker, start_date, end_date)
    await pg_conn.load_data(table_name, ticker, start_date, end_date_1)
    # print(df_postgres.head())
    await pg_conn.close()


if __name__ == '__main__':
    asyncio.run(main())


# if __name__ == '__main__':
#     parquet_path = 'data/eod_data.parquet'
#     db_url = os.getenv("POSTGRES_CNX")
#     table_name = 'eods'
#     ticker = 'RGTI'
#     start_date = '2023-01-01'
#     end_date = '2023-12-31'
#     end_date_1 = '2024-01-31'

#     print("Loading EOD data from Parquet file...")
#     df_parquet = load_from_parquet(ticker, start_date, end_date)
#     df_parquet = load_from_parquet(ticker, start_date, end_date_1)
#     # print(df_parquet.head())

#     print("\nLoading EOD data from PostgreSQL database using Polars...")
#     df_postgres = load_from_postgres_polars(db_url, table_name, ticker, start_date, end_date)
#     df_postgres = load_from_postgres_polars(db_url, table_name, ticker, start_date, end_date_1)
    # print(df_postgres.head())


# if __name__ == '__main__':
#     parquet_path = 'data/eod_data.parquet'
#     db_url = os.getenv("POSTGRES_CNX")
#     table_name = 'eods'
#     ticker = 'MSTR'
#     start_date = '2023-01-01'
#     end_date = '2023-12-31'

#     print("Loading EOD data from Parquet file...")
#     df_parquet = load_from_parquet(ticker, start_date, end_date)
#     print(df_parquet.head())

#     print("\nLoading EOD data from PostgreSQL database...")
#     asyncio.run(load_from_postgres(db_url, table_name, ticker, start_date, end_date))
