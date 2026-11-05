import requests
import pandas as pd


class BybitDataLoader:
    """Загружает публичные свечи Bybit для USDT-бессрочных контрактов."""

    BASE_URL = "https://api.bybit.com/v5/market/kline"
    INTERVALS = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15",
        "30m": "30", "1h": "60", "2h": "120", "4h": "240",
        "6h": "360", "12h": "720", "1d": "D", "1w": "W", "1M": "M",
    }

    def __init__(self, symbol: str, interval: str, limit: int = 1000):
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit

    def fetch(self) -> pd.DataFrame:
        if self.interval not in self.INTERVALS:
            raise ValueError(f"Неподдерживаемый таймфрейм Bybit: {self.interval}")
        if self.limit < 1:
            raise ValueError("Количество свечей должно быть положительным.")

        rows = []
        end_time = None
        while len(rows) < self.limit:
            page_size = min(1000, self.limit - len(rows))
            params = {
                "category": "linear",
                "symbol": self.symbol,
                "interval": self.INTERVALS[self.interval],
                "limit": page_size,
            }
            if end_time is not None:
                params["end"] = end_time

            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if payload.get("retCode") != 0:
                raise ValueError(f"Ошибка Bybit API: {payload.get('retMsg')}")

            page = payload["result"]["list"]
            if not page:
                break
            rows.extend(page)
            oldest_time = min(int(row[0]) for row in page)
            end_time = oldest_time - 1
            if len(page) < page_size:
                break

        if not rows:
            raise ValueError(f"Bybit не вернул свечи для {self.symbol} ({self.interval}).")

        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume", "quote_volume"
        ])
        df = df.drop_duplicates(subset="open_time")
        df = df.sort_values("open_time").tail(self.limit).reset_index(drop=True)
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
        df[numeric] = df[numeric].astype(float)
        df["close_time"] = (df["open_time"].astype("int64") // 10**6) + self._interval_ms(self.interval) - 1
        df["trades"] = 0
        # У Bybit Kline endpoint нет разбивки объёма на taker buy/sell.
        # Не подставляем фиктивные значения: признак будет исключён из модели.
        df["taker_buy_volume"] = float("nan")
        df["taker_buy_quote_volume"] = float("nan")
        df["ignore"] = 0
        return df.sort_values("open_time").reset_index(drop=True)

    @classmethod
    def _interval_ms(cls, interval: str) -> int:
        value = cls.INTERVALS[interval]
        if value == "D":
            return 24 * 60 * 60 * 1000
        if value == "W":
            return 7 * 24 * 60 * 60 * 1000
        if value == "M":
            return 31 * 24 * 60 * 60 * 1000
        return int(value) * 60 * 1000
