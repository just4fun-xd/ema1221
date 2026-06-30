"""Погодовой прогон Калман-версии (динамическая β) vs статичная.

Запуск: python run_pairs_kalman.py

Прямой ответ на риск #1 Karpathy-council ("β look-ahead, rolling-β
перед сдачей"). Сравниваем бок о бок: статичная OLS-β (look-ahead) и
Калман-β (online, без заглядывания) на одних парах и годах.

ВАЖНО: Калман — НЕ гарантированное улучшение. На синтетике с резким
сломом β он временно проигрывает, пока адаптируется. Реальный ответ
даёт только этот прогон. Калман чувствителен к delta/R — параметры
явные, не подобраны молча.
"""

import pandas as pd
import statsmodels.api as sm

from engine import load_data, run_engine
from display import print_row
from strategies_pairs import zscore_spread
from strategies_pairs_kalman import zscore_spread_kalman

cost = 0.001

periods = [
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

# Corn/Wheat — единственная из трёх вела себя как "слабый, но не
# сломанный" сигнал. Если что-то оживёт от rolling-β, то она.
selected = [
    ("Corn", "ZC=F", "Wheat", "ZW=F", 38),
    ("Crude", "CL=F", "Gasoline", "RB=F", 11),
]


def fit_beta(close_a, close_b):
    df = pd.concat([close_a, close_b], axis=1,
                   keys=["A", "B"], sort=True).dropna()
    x = sm.add_constant(df["B"])
    return sm.OLS(df["A"], x).fit().params.iloc[1]


def run_both(ticker_a, ticker_b, window, trade_start, end):
    """Возвращает (static_ret, static_dd, kalman_ret, kalman_dd)."""
    load_start = (
        pd.Timestamp(trade_start) - pd.DateOffset(years=3)
    ).strftime("%Y-%m-%d")
    close_a = load_data(ticker_a, load_start, end)
    close_b = load_data(ticker_b, load_start, end)

    beta = fit_beta(close_a, close_b)
    sp_s, pos_s = zscore_spread(close_a, close_b, beta=beta, window=window)
    _, ret_s, dd_s = run_engine(sp_s, pos_s, trade_start, cost)

    sp_k, pos_k = zscore_spread_kalman(close_a, close_b, window=window)
    _, ret_k, dd_k = run_engine(sp_k, pos_k, trade_start, cost)

    return ret_s, dd_s, ret_k, dd_k


for name_a, ticker_a, name_b, ticker_b, window in selected:
    print(f"\n=== {name_a}/{name_b} (window={window}) ===")
    print(f"{'Year':<8}{'StaticRet':>11}{'StaticDD':>10}"
          f"{'KalmanRet':>12}{'KalmanDD':>10}")
    for trade_start, end in periods:
        ret_s, dd_s, ret_k, dd_k = run_both(
            ticker_a, ticker_b, window, trade_start, end)
        print(f"{trade_start[:4]:<8}"
              f"{ret_s*100:>10.1f}%{dd_s*100:>9.1f}%"
              f"{ret_k*100:>11.1f}%{dd_k*100:>9.1f}%")