"""Прогон z-score спред-стратегии на выбранных коинтегрированных парах.

Запуск: python run_pairs.py

Контекст: mean-reversion на сырье возможна только через спред
коинтегрированных пар (одиночное сырьё трендит). β считается тем же
OLS, что и в pair_diagnostic.py — статичная, look-ahead, POC only.
Окно z-score = half-life пары (из диагностики).
"""

import pandas as pd
import statsmodels.api as sm

from engine import load_data, run_engine
from display import print_row
from strategies_pairs import zscore_spread

cost = 0.001
load_start = "2015-01-01"
trade_start = "2018-01-01"   # 3 года на прогрев z-окна + β
end = "2025-01-01"

# (имя A, тикер A, имя B, тикер B, window=half-life из диагностики)
selected = [
    ("Crude", "CL=F", "Gasoline", "RB=F", 11),
    ("Corn", "ZC=F", "Wheat", "ZW=F", 38),
    ("Crude", "CL=F", "Brent", "BZ=F", 3),   # стресс-тест комиссий
]


def fit_beta(ticker_a, ticker_b, start, end):
    """Hedge ratio через OLS A = alpha + beta*B (как в диагностике)."""
    a = load_data(ticker_a, start, end)
    b = load_data(ticker_b, start, end)
    df = pd.concat([a, b], axis=1, keys=["A", "B"], sort=True).dropna()
    x = sm.add_constant(df["B"])
    model = sm.OLS(df["A"], x).fit()
    return model.params.iloc[1]


print(f"\n{'Pair':<22}{'Return':>10}{'Drawdown':>10}")
print("-" * 42)

for name_a, ticker_a, name_b, ticker_b, window in selected:
    beta = fit_beta(ticker_a, ticker_b, load_start, end)

    close_a = load_data(ticker_a, load_start, end)
    close_b = load_data(ticker_b, load_start, end)

    synth_price, position = zscore_spread(
        close_a, close_b, beta=beta, window=window,
        entry=2.0, exit=0.5)

    _, total_return, max_dd = run_engine(
        synth_price, position, trade_start, cost)

    label = f"{name_a}/{name_b} (w={window})"
    print_row(label, total_return, max_dd, col_width=22)