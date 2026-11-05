import argparse
from datetime import datetime, timezone

from src.data_loader import BybitDataLoader
from src.market_database import (
    get_latest_open_time_ms,
    remove_unclosed_candles,
    save_candles,
)


BYBIT_CONTRACTS = [
    "BTCUSDT",
    "SOXSUSDT",
    "SOXLUSDT",
    "1000PEPEUSDT",
    "PUMPFUNUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "NEARUSDT",
    "CAPUSDT",
    "WUSDT",
    "SUSDT",
]
INTERVALS = ["1h", "15m", "4h"]
HISTORY_SIZE = 10_000
INTERVAL_MS = {"15m": 15 * 60_000, "1h": 60 * 60_000, "4h": 4 * 60 * 60_000}


def main():
    parser = argparse.ArgumentParser(description="Загрузка и обновление локальной базы свечей Bybit.")
    parser.add_argument("--interval", choices=INTERVALS, help="Обработать только выбранный таймфрейм.")
    parser.add_argument("--full", action="store_true", help="Заново загрузить полную историю для выбранных интервалов.")
    args = parser.parse_args()
    intervals = [args.interval] if args.interval else INTERVALS

    print(
        f"Обновляю Bybit: {len(BYBIT_CONTRACTS)} контрактов, "
        f"таймфреймы {', '.join(intervals)}, до {HISTORY_SIZE} свечей на пару"
    )
    failed = []
    for symbol in BYBIT_CONTRACTS:
        for interval in intervals:
            try:
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                removed_open = remove_unclosed_candles("bybit", symbol, interval, now_ms)
                latest_ms = None if args.full else get_latest_open_time_ms("bybit", symbol, interval)
                if latest_ms is None:
                    candles = BybitDataLoader(symbol, interval, limit=HISTORY_SIZE).fetch()
                    print(f"{symbol} {interval}: загружаю начальную историю")
                else:
                    # Запрос начинаем строго со следующей свечи после последней закрытой.
                    candles = BybitDataLoader(symbol, interval, limit=1000).fetch(
                        start_time_ms=latest_ms + INTERVAL_MS[interval]
                    )
                close_times = candles["close_time"].astype("int64")
                candles = candles[close_times < now_ms].reset_index(drop=True)
                if candles.empty:
                    print(f"{symbol} {interval}: новых закрытых свечей нет (удалено незакрытых: {removed_open})")
                    continue
                count = save_candles("bybit", symbol, interval, candles)
                print(f"{symbol} {interval}: добавлено {count} новых закрытых свечей")
            except Exception as error:
                failed.append((symbol, interval, str(error)))
                print(f"{symbol} {interval}: ошибка — {error}")

    print("\nБаза данных: market_data.sqlite")
    if failed:
        print("Не обновились:")
        for symbol, interval, error in failed:
            print(f"- {symbol} {interval}: {error}")


if __name__ == "__main__":
    main()
