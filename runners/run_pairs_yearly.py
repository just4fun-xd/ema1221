"""Погодовой прогон z-score спред-стратегии (2021-2025).

Запуск: python run_pairs_yearly.py

Один агрегат за весь период прячет, ГДЕ стратегия течёт. Разбивка по
годам показывает: равномерно слабая или один год всё убил. Принцип
"один год ничего не доказывает" — поэтому смотрим все пять сразу.

β статичная, считается на загруженном окне (3 года до года торговли) —
POC look-ahead, ОК для фальсификатора, rolling-β перед сдачей.
"""

import pandas as pd
import statsmodels.api as sm

from core.engine import load_data, run_engine
from core.display import print_row
from strategies.strategies_pairs import zscore_spread

cost = 0.001

periods = [
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

# (имя A, тикер A, имя B, тикер B, window=half-life из диагностики)
selected = [
    ("Crude", "CL=F", "Gasoline", "RB=F", 11),
    ("Corn", "ZC=F", "Wheat", "ZW=F", 38),
    ("Crude", "CL=F", "Brent", "BZ=F", 3),
]


def fit_beta(close_a, close_b):
    """Hedge ratio через OLS A = alpha + beta*B (как в диагностике)."""
    df = pd.concat([close_a, close_b], axis=1,
                   keys=["A", "B"], sort=True).dropna()
    x = sm.add_constant(df["B"])
    return sm.OLS(df["A"], x).fit().params.iloc[1]


def run_pair_year(ticker_a, ticker_b, window, trade_start, end):
    """Один год, одна пара: грузим с прогревом, считаем β, гоняем."""
    load_start = (
        pd.Timestamp(trade_start) - pd.DateOffset(years=3)
    ).strftime("%Y-%m-%d")

    close_a = load_data(ticker_a, load_start, end)
    close_b = load_data(ticker_b, load_start, end)

    beta = fit_beta(close_a, close_b)
    synth_price, position = zscore_spread(
        close_a, close_b, beta=beta, window=window,
        entry=2.0, exit=0.5)

    _, total_return, max_dd = run_engine(
        synth_price, position, trade_start, cost)
    return total_return, max_dd


for name_a, ticker_a, name_b, ticker_b, window in selected:
    print(f"\n=== {name_a}/{name_b} (window={window}) ===")
    print(f"{'Year':<14}{'Return':>10}{'Drawdown':>10}")
    for trade_start, end in periods:
        ret, dd = run_pair_year(
            ticker_a, ticker_b, window, trade_start, end)
        print_row(trade_start[:4], ret, dd)