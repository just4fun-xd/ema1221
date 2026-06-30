"""Per-instrument / per-year report for dual_momentum_tilt.

Old dev_test.py output format. Uses run_portfolio_yearly, which isolates
each instrument's standalone P&L given its weight in the tilt matrix.

READ THIS before interpreting:
  This shows each instrument's contribution AS WEIGHTED BY THE STRATEGY,
  not the instrument's own raw trend. A cross-sectional strategy shorts
  weak names and longs strong ones, so a column can be NEGATIVE return
  even in a year the stock rose -- because tilt was SHORT it. Likewise
  many 0.0% cells are correct: that name was neither top nor bottom
  quintile that month, so weight was zero. This is the diagnostic value:
  it shows WHERE the legs land, not a per-stock buy-and-hold.
"""

import pandas as pd

from core.engine import load_data
from core.engine_portfolio import load_basket, run_portfolio_yearly
from strategies.strategies_dualmom import dual_momentum_tilt

BASKET = {
    "S&P 500": "SPY", "Nasdaq": "QQQ", "Apple": "AAPL", "Tesla": "TSLA",
    "Nvidia": "NVDA", "Microsoft": "MSFT", "Amazon": "AMZN", "Meta": "META",
    "Alphabet": "GOOGL", "AMD": "AMD", "Netflix": "NFLX", "Visa": "V",
    "Mastercard": "MA", "UnitedHealth": "UNH", "Home Depot": "HD",
    "Bank of America": "BAC", "Coca-Cola": "KO", "Procter & Gamble": "PG",
    "Walmart": "WMT",
}

LOAD_START = "2017-01-01"
END = "2026-01-01"
COST = 0.001

PERIODS = [
    ("2019-01-01", "2020-01-01"),
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]


def main():
    prices = load_basket(BASKET, LOAD_START, END)
    spy = load_data("SPY", LOAD_START, END)
    weights = dual_momentum_tilt(prices, benchmark=spy)

    print("\n" + "=" * 50)
    print("=== Group: Diversified Portfolio ===")
    print("=== Strategy: Dual mom tilt ===")
    print("=== (negative = tilt was SHORT; 0.0% = not in any leg) ===")
    run_portfolio_yearly(prices, weights, PERIODS, COST)


if __name__ == "__main__":
    main()