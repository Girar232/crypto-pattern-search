from pathlib import Path
import sqlite3

import pandas as pd


DATABASE_PATH = Path(__file__).resolve().parent.parent / "market_data.sqlite"


def get_symbols(exchange: str) -> list[str]:
    if not DATABASE_PATH.exists():
        return []
    with sqlite3.connect(DATABASE_PATH) as connection:
        try:
            rows = connection.execute(
                "SELECT DISTINCT symbol FROM candles WHERE exchange = ? ORDER BY symbol",
                (exchange,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [row[0] for row in rows]


def get_intervals(exchange: str, symbol: str) -> list[str]:
    if not DATABASE_PATH.exists():
        return []
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            "SELECT DISTINCT interval FROM candles "
            "WHERE exchange = ? AND symbol = ? "
            "ORDER BY CASE interval WHEN '1h' THEN 0 WHEN '15m' THEN 1 WHEN '4h' THEN 2 ELSE 3 END, interval",
            (exchange, symbol),
        ).fetchall()
    return [row[0] for row in rows]


def get_latest_open_time_ms(exchange: str, symbol: str, interval: str) -> int | None:
    if not DATABASE_PATH.exists():
        return None
    with sqlite3.connect(DATABASE_PATH) as connection:
        try:
            row = connection.execute(
                "SELECT MAX(open_time) FROM candles WHERE exchange = ? AND symbol = ? AND interval = ?",
                (exchange, symbol, interval),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    if not row or row[0] is None:
        return None
    return int(pd.Timestamp(row[0]).timestamp() * 1000)


def remove_unclosed_candles(exchange: str, symbol: str, interval: str, now_ms: int) -> int:
    """Удаляет только текущие незакрытые свечи, чтобы хранить закрытую историю неизменной."""
    if not DATABASE_PATH.exists():
        return 0
    with sqlite3.connect(DATABASE_PATH) as connection:
        try:
            cursor = connection.execute(
                "DELETE FROM candles WHERE exchange = ? AND symbol = ? AND interval = ? "
                "AND close_time >= ?",
                (exchange, symbol, interval, now_ms),
            )
        except sqlite3.OperationalError:
            return 0
        return cursor.rowcount


def read_candles(exchange: str, symbol: str, interval: str) -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        raise ValueError("База данных ещё не создана. Запусти update_database.py на компьютере.")
    with sqlite3.connect(DATABASE_PATH) as connection:
        df = pd.read_sql_query(
            "SELECT open_time, open, high, low, close, volume, close_time, "
            "quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore "
            "FROM candles WHERE exchange = ? AND symbol = ? AND interval = ? "
            "ORDER BY open_time",
            connection,
            params=(exchange, symbol, interval),
        )
    if df.empty:
        raise ValueError(f"В базе нет свечей для {symbol} ({interval}).")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def save_candles(exchange: str, symbol: str, interval: str, df: pd.DataFrame) -> int:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "exchange", "symbol", "interval", "open_time", "open", "high", "low",
        "close", "volume", "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]
    frame = df.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    frame["exchange"] = exchange
    frame["symbol"] = symbol
    frame["interval"] = interval
    for column in columns:
        if column not in frame:
            frame[column] = None

    records = frame[columns].itertuples(index=False, name=None)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS candles (
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                close_time INTEGER, quote_volume REAL, trades INTEGER,
                taker_buy_volume REAL, taker_buy_quote_volume REAL, ignore REAL,
                PRIMARY KEY (exchange, symbol, interval, open_time)
            )"""
        )
        cursor = connection.executemany(
            "INSERT OR IGNORE INTO candles (" + ",".join(columns) + ") VALUES (" +
            ",".join("?" for _ in columns) + ")",
            records,
        )
        inserted = cursor.rowcount
    return inserted
