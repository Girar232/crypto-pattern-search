from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import BybitDataLoader
from src.feature_engineering import build_feature_matrix
from src.similarity import find_similar


st.set_page_config(
    page_title="Поиск похожих ситуаций",
    page_icon="📈",
    layout="wide",
)

st.title("Похожие рыночные ситуации")
st.caption(
    "Источник данных: Bybit USDT-бессрочные контракты (linear). "
    "Исторические результаты не гарантируют будущее движение цены."
)

with st.sidebar:
    st.header("Параметры анализа")
    symbol = st.text_input("Контракт Bybit linear", "BTCUSDT").strip().upper()
    interval = st.selectbox("Таймфрейм", ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"], index=5)
    history_size = st.selectbox("Свечей загрузить", [1000, 2000, 5000, 10000], index=0)
    window_size = st.selectbox("Длина окна, свечей", [20, 40, 60], index=0)
    neighbor_count = st.slider("Похожих эпизодов", 3, 20, 10)
    horizon = st.slider("Горизонт результата, свечей", 1, 48, 10)
    run = st.button("Загрузить и проанализировать", type="primary", use_container_width=True)

if run:
    st.session_state["analysis_requested"] = True

if st.session_state.get("analysis_requested", False):
    if not symbol:
        st.error("Укажи инструмент, например BTCUSDT.")
        st.stop()

    try:
        with st.spinner("Загружаю свечи Bybit и ищу похожие эпизоды…"):
            candles = BybitDataLoader(
                symbol=symbol,
                interval=interval,
                limit=history_size,
            ).fetch()

            # Не включаем последнюю свечу, пока она ещё формируется.
            now = datetime.now(timezone.utc)
            close_times = pd.to_numeric(candles["close_time"], errors="coerce")
            candles = candles[close_times < int(now.timestamp() * 1000)].reset_index(drop=True)

            features = build_feature_matrix(candles, window_size=window_size)
            results = find_similar(
                candles=candles,
                features=features,
                n_neighbors=neighbor_count,
                horizon=horizon,
                window_size=window_size,
            )

        avg_return = results["future_return"].mean()
        median_return = results["future_return"].median()
        positive_share = (results["future_return"] > 0).mean()
        avg_distance = results["distance"].mean()
        baseline = (candles["close"].shift(-horizon) / candles["close"] - 1).dropna()

        st.subheader("Сравнение результатов")
        comparison = pd.DataFrame([
            {
                "Выборка": "Похожие ситуации",
                "Средняя доходность": f"{avg_return:.2%}",
                "Медианная доходность": f"{median_return:.2%}",
                "Периодов с ростом": f"{positive_share:.0%}",
                "Средняя дистанция": f"{avg_distance:.3f}",
            },
            {
                "Выборка": "Все периоды (база)",
                "Средняя доходность": f"{baseline.mean():.2%}",
                "Медианная доходность": f"{baseline.median():.2%}",
                "Периодов с ростом": f"{(baseline > 0).mean():.0%}",
                "Средняя дистанция": "—",
            },
        ])
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Выборка": st.column_config.TextColumn(
                    help="Какие периоды собраны в этой строке: выбранные похожие окна или все доступные периоды для базы."
                ),
                "Средняя доходность": st.column_config.TextColumn(
                    help="Среднее изменение цены за выбранный горизонт."
                ),
                "Медианная доходность": st.column_config.TextColumn(
                    help="Изменение цены в середине распределения; меньше зависит от крайних скачков."
                ),
                "Периодов с ростом": st.column_config.TextColumn(
                    help="Доля периодов, в которых цена выросла за выбранный горизонт."
                ),
                "Средняя дистанция": st.column_config.TextColumn(
                    help="Среднее расстояние по признакам от текущего окна до выбранных аналогов. Меньше — ближе; это не вероятность. Для базы не рассчитывается."
                ),
            },
        )

        st.subheader(f"{symbol} · {interval}: все похожие эпизоды")
        current = features.iloc[-1]
        current_start = pd.Timestamp(current["start_time"])
        time_to_index = {timestamp: i for i, timestamp in enumerate(candles["open_time"])}
        target_idx = time_to_index[current_start]
        current_candles = candles.iloc[target_idx:target_idx + window_size]
        current_prices = current_candles["close"].to_numpy()

        def normalized_path(start_time):
            start_idx = time_to_index[pd.Timestamp(start_time)]
            end_idx = start_idx + window_size + horizon
            prices = candles.iloc[start_idx:end_idx]["close"].to_numpy()
            return prices / prices[0] * 100

        def add_path(chart, name, values, color=None, width=2):
            chart.add_trace(go.Scatter(
                x=list(range(len(values))), y=values, mode="lines", name=name,
                line={"color": color, "width": width} if color else {"width": width},
            ))

        overview = go.Figure()

        def add_candles(chart, name, start_idx, color):
            candle_count = window_size if name == "Текущая ситуация" else window_size + horizon
            part = candles.iloc[start_idx:start_idx + candle_count]
            base = part.iloc[0]["open"]
            is_current = name == "Текущая ситуация"
            chart.add_trace(go.Candlestick(
                x=list(range(candle_count)),
                open=part["open"] / base * 100,
                high=part["high"] / base * 100,
                low=part["low"] / base * 100,
                close=part["close"] / base * 100,
                customdata=part["open_time"].dt.strftime("%d.%m.%Y %H:%M UTC"),
                name=name,
                increasing_line_color="#111111" if is_current else "#16a34a",
                decreasing_line_color="#111111" if is_current else "#dc2626",
                increasing_fillcolor="#ffffff" if is_current else "#16a34a",
                decreasing_fillcolor="#111111" if is_current else "#dc2626",
                visible="legendonly" if name.startswith("Аналог") else True,
                hovertemplate=(
                    "Открытие свечи: %{customdata}<br>"
                    "Свеча от начала окна: %{x}<br>"
                    "Open: %{open:.2f}<br>High: %{high:.2f}<br>"
                    "Low: %{low:.2f}<br>Close: %{close:.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ))

        for number, (_, row) in enumerate(results.iterrows(), start=1):
            start_idx = time_to_index[pd.Timestamp(row["start_time"])]
            add_candles(overview, f"Аналог {number} · доходность {row['future_return']:.2%} · дистанция {row['distance']:.3f}", start_idx, "#4c78a8")
        add_candles(overview, "Текущая ситуация", target_idx, "#222222")
        overview.update_xaxes(rangeslider_visible=False)
        overview.add_vline(x=window_size - 0.5, line_dash="dash", line_color="royalblue")
        overview.add_hline(y=100, line_dash="dot", line_color="gray")
        overview.update_layout(
            xaxis_title="Свечи от начала окна",
            yaxis_title="Цена (начало каждого окна = 100)",
            legend_title="Исторический результат",
            height=520,
        )
        st.plotly_chart(overview, use_container_width=True)
        st.caption(f"Вертикальная линия отделяет исходное окно из {window_size} свечей от последующих {horizon} свечей. Будущее движение текущей ситуации неизвестно.")
        st.subheader("Исторические аналоги и их последующий результат")
        display = results[["start_time", "end_time", "distance", "future_return"]].copy()
        display["start_time"] = pd.to_datetime(display["start_time"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        display["end_time"] = pd.to_datetime(display["end_time"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        display["future_return"] = display["future_return"].map(lambda value: f"{value:.2%}")
        display["distance"] = display["distance"].map(lambda value: f"{value:.3f}")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "start_time": st.column_config.TextColumn(
                    "Начало окна",
                    help="Время открытия первой свечи исторического аналога."
                ),
                "end_time": st.column_config.TextColumn(
                    "Конец окна",
                    help="Время открытия последней свечи исходного окна."
                ),
                "distance": st.column_config.TextColumn(
                    "Дистанция",
                    help="Насколько отличаются признаки исторического окна от текущего. Чем меньше, тем ближе; это не вероятность."
                ),
                "future_return": st.column_config.TextColumn(
                    "Доходность после окна",
                    help="Изменение цены за выбранное число свечей после исторического окна."
                ),
            },
        )

    except Exception as error:
        st.error(f"Не удалось выполнить анализ: {error}")
else:
    st.info("Выбери параметры слева и нажми кнопку, чтобы получить свежие данные Bybit.")
