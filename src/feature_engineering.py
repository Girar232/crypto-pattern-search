import numpy as np
import pandas as pd


WINDOW_SIZE = 20
STEP = 1
VOLUME_LOOKBACK = 100


def calculate_features(
    window: pd.DataFrame,
    previous_volume: float
) -> dict:
    """
    Рассчитывает 12 признаков одного рыночного окна.
    """

    open_prices = window["open"].to_numpy()
    close_prices = window["close"].to_numpy()
    high_prices = window["high"].to_numpy()
    low_prices = window["low"].to_numpy()
    volumes = window["volume"].to_numpy()

    # 1. Общая доходность за 20 свечей
    total_return = close_prices[-1] / open_prices[0] - 1

    # 2. Волатильность часовых доходностей
    hourly_returns = np.diff(
        np.concatenate(([open_prices[0]], close_prices))
    ) / np.concatenate(([open_prices[0]], close_prices))[:-1]

    volatility = np.std(hourly_returns)

    # 3. Максимальная просадка от локального максимума
    running_max = np.maximum.accumulate(close_prices)
    drawdowns = close_prices / running_max - 1
    max_drawdown = abs(drawdowns.min())

    # 4. Максимальный рост от локального минимума
    running_min = np.minimum.accumulate(close_prices)
    rallies = close_prices / running_min - 1
    max_rally = rallies.max()

    # 5. Средний объём относительно предыдущих 100 свечей
    avg_volume_ratio = volumes.mean() / previous_volume

    # 6. Изменение объёма: последние 5 часов против первых 5
    volume_trend = (
        volumes[-5:].mean() / volumes[:5].mean() - 1
    )

    # Средний диапазон свечи относительно цены открытия
    candle_range = np.mean(
        (high_prices - low_prices) / open_prices
    )

    # 9–12. Доходность четырёх равных частей окна
    segment_returns = []
    segment_size = len(window) // 4

    for i in range(0, len(window), segment_size):
        segment = window.iloc[i:i + segment_size]

        segment_return = (
            segment["close"].iloc[-1] /
            segment["open"].iloc[0] - 1
        )

        segment_returns.append(segment_return)

    return {
        "total_return": total_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "max_rally": max_rally,
        "avg_volume_ratio": avg_volume_ratio,
        "volume_trend": volume_trend,
        "candle_range": candle_range,
        "return_1": segment_returns[0],
        "return_2": segment_returns[1],
        "return_3": segment_returns[2],
        "return_4": segment_returns[3]
    }


def build_feature_matrix(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
    volume_lookback: int = VOLUME_LOOKBACK
) -> pd.DataFrame:
    """
    Создаёт матрицу признаков для всех рыночных окон.
    """

    if window_size < 4 or window_size % 4 != 0:
        raise ValueError("Размер окна должен быть положительным числом, кратным 4.")

    if step < 1:
        raise ValueError("Шаг должен быть положительным.")

    if volume_lookback < 1:
        raise ValueError("Период расчёта объёма должен быть положительным.")

    required_columns = {
        "open_time", "open", "high", "low", "close", "volume"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Отсутствуют столбцы: {missing_columns}")

    if len(df) < volume_lookback + window_size:
        raise ValueError("Недостаточно исторических данных.")

    if (df["volume"] <= 0).any():
        raise ValueError("Обнаружен нулевой или отрицательный объём.")

    if (df["open"] <= 0).any():
        raise ValueError("Обнаружена некорректная цена открытия.")

    results = []

    for start in range(
        volume_lookback,
        len(df) - window_size + 1,
        step
    ):
        window = df.iloc[start:start + window_size]

        previous_volume = df["volume"].iloc[
            start - volume_lookback:start
        ].mean()

        features = calculate_features(
            window,
            previous_volume
        )
        if "taker_buy_volume" in window.columns and window["taker_buy_volume"].notna().all():
            features["buy_pressure"] = (
                window["taker_buy_volume"].sum() / window["volume"].sum()
            )

        features["start_time"] = window["open_time"].iloc[0]
        features["end_time"] = window["open_time"].iloc[-1]

        results.append(features)

    return pd.DataFrame(results)
