"""Compare cross-sectional and per-series strategies on ONE basket.

Cross-sectional strategies (dual_momentum*) need the whole basket and
run natively through run_portfolio. Per-series strategies (ema_cross,
donchian_*) are wrapped by positions_to_weights so they run through the
SAME portfolio engine -- apples-to-apples on one P&L, one cost model.

This is the correct way to put dual_momentum_tilt next to Donchian/EMA.
Do NOT drop tilt into the run_engine per-series loop: it gets one
instrument at a time and cannot rank a basket of one.
"""

import pandas as pd

from engine import load_data
from strategies import ema_cross, ema_ensemble_voltarget
from engine_portfolio import (load_basket, positions_to_weights,
                              run_portfolio, portfolio_beta)
from strategies_dualmom import dual_momentum_tilt

# Your per-series strategies to compare (add donchian_* imports as needed):
#   from not_now import donchian_ensemble_macd_voltarget, ...
PERSERIES = {
    "EMA Cross": ema_cross,
    "EMA Ens+VT": ema_ensemble_voltarget,
    # "Donchian MACD Vol": donchian_ensemble_macd_voltarget,
    # "Donchian MACD Pyr": donchian_ensemble_macd_pyramid,
}

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
TRADE_START = "2018-07-01"
COST = 0.001


def main():
    prices = load_basket(BASKET, LOAD_START, END)
    spy = load_data("SPY", LOAD_START, END)
    print(f"Loaded {prices.shape[1]} instruments, {prices.shape[0]} rows\n")

    rows = []

    # Per-series strategies via the adapter.
    for name, fn in PERSERIES.items():
        w = positions_to_weights(prices, fn, gross=1.0)
        _, r, dd = run_portfolio(prices, w, TRADE_START, COST,
                                 target_vol=0.10)
        b = portfolio_beta(prices, w, spy, TRADE_START)
        rows.append((name, r, dd, b, "per-series"))

    # Cross-sectional tilt, native.
    w_tilt = dual_momentum_tilt(prices, benchmark=spy)
    _, r, dd = run_portfolio(prices, w_tilt, TRADE_START, COST,
                             target_vol=0.10)
    b = portfolio_beta(prices, w_tilt, spy, TRADE_START)
    rows.append(("Dual mom tilt", r, dd, b, "cross-sec"))

    print("=" * 64)
    print("  All strategies as ONE portfolio (target_vol=0.10)")
    print("=" * 64)
    print(f"{'Strategy':<18}{'TotRet':>10}{'MaxDD':>10}{'Beta':>9}{'Type':>13}")
    print("-" * 64)
    for name, r, dd, b, kind in rows:
        print(f"{name:<18}{r*100:>9.1f}%{dd*100:>9.1f}%{b:>+9.2f}{kind:>13}")
    print("-" * 64)
    print("  Same basket, same engine, same costs -> comparable.")


if __name__ == "__main__":
    main()