import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


FEATURE_COLUMNS = [
    "total_return",
    "volatility",
    "max_drawdown",
    "max_rally",
    "avg_volume_ratio",
    "volume_trend",
    "candle_range",
    "return_1",
    "return_2",
    "return_3",
    "return_4"
]


def find_similar(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    n_neighbors: int = 10,
    horizon: int = 10,
    window_size: int = 20
) -> pd.DataFrame:
    """
    Ищет похожие исторические ситуации без пересечения окон.
    """

    if n_neighbors < 1 or horizon < 1 or window_size < 1:
        raise ValueError("Параметры должны быть положительными.")

    target = features.iloc[-1]
    target_start_idx = candles.index[
        candles["open_time"] == target["start_time"]
    ].tolist()

    if not target_start_idx:
        raise ValueError("Не удалось найти начало текущего окна в свечах.")

    target_start_idx = target_start_idx[0]

    # Исключаем последнее окно — оно является целевым.
    candidates = features.iloc[:-1].copy()

    # Определяем позиции начала и конца каждого окна в свечах.
    time_to_index = {
        timestamp: index
        for index, timestamp in enumerate(candles["open_time"])
    }

    candidates["start_idx"] = candidates["start_time"].map(time_to_index)
    candidates["end_idx"] = candidates["end_time"].map(time_to_index)

    # Исключаем ситуации, для которых ещё нет полного будущего горизонта.
    candidates = candidates[
        candidates["end_idx"] + horizon < target_start_idx
    ].copy()

    if candidates.empty:
        raise ValueError("Недостаточно исторических ситуаций.")

    # Масштабируем признаки.
    X_history = candidates[FEATURE_COLUMNS]
    X_target = target[FEATURE_COLUMNS].to_frame().T

    scaler = StandardScaler()

    X_history_scaled = scaler.fit_transform(X_history)
    X_target_scaled = scaler.transform(X_target)

    # Рассчитываем расстояния до всех исторических ситуаций.
    model = NearestNeighbors(
        n_neighbors=len(candidates),
        metric="euclidean"
    )

    model.fit(X_history_scaled)

    distances, indices = model.kneighbors(X_target_scaled)

    # Сортируем кандидатов от наиболее похожих к менее похожим.
    ordered_candidates = candidates.iloc[indices[0]].copy()
    ordered_candidates["distance"] = distances[0]

    # Отбираем ситуации, которые не пересекаются друг с другом.
    selected = []

    for _, row in ordered_candidates.iterrows():

        overlaps = any(
            abs(row["start_idx"] - item["start_idx"]) < window_size + horizon
            for item in selected
        )

        if overlaps:
            continue

        selected.append(row)

        if len(selected) == n_neighbors:
            break

    if not selected:
        raise ValueError("Не удалось найти непересекающиеся ситуации.")

    results = pd.DataFrame(selected)

    # Рассчитываем доходность после каждой найденной ситуации.
    future_returns = []

    for _, row in results.iterrows():

        end_idx = int(row["end_idx"])
        future_idx = end_idx + horizon

        start_price = candles.iloc[end_idx]["close"]
        future_price = candles.iloc[future_idx]["close"]

        future_return = future_price / start_price - 1

        future_returns.append(future_return)

    results["future_return"] = future_returns

    return results.drop(
        columns=["start_idx", "end_idx"]
    ).reset_index(drop=True)
