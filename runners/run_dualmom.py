"""Cross-sectional dual momentum -- validation + variant comparison.

Lives alongside dev_test.py / main.py, does NOT replace them. Uses the
project's own load_data via engine_portfolio.load_basket, so on your
machine it runs unchanged against real yfinance data.

Outputs:
  A) Variant comparison: base + 3 improvements, each scored on the three
     honesty tests from SHORT_REDESIGN.md (beta~0, 2022 spread, DD<40%).
  B) Per-instrument, per-year report (old dev_test.py format) for one
     chosen variant.
  C) Sanity check that run_portfolio reproduces run_engine.
"""

import pandas as pd

from core.engine import load_data, run_engine
from strategies.strategies import (ema_ensemble, cross_sectional_dual_momentum)
from core.engine_portfolio import (load_basket, positions_to_weights,
                              run_portfolio, portfolio_beta,
                              run_portfolio_yearly)
from strategies.strategies_dualmom import (dual_momentum, dual_momentum_long_only,
                                dual_momentum_tilt, dual_momentum_regime,
                                dual_momentum_volscaled)


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

PERIODS = [
    ("2019-01-01", "2020-01-01"),
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]


def yearly(result):
    out = {}
    for y, g in result.groupby(result.index.year):
        r = g["equity"].iloc[-1] / g["equity"].iloc[0] - 1
        out[y] = (r, g["drawdown"].min())
    return out


def sanity_check(prices):
    col = prices.columns[0]
    one = prices[[col]]
    w = positions_to_weights(one, ema_ensemble, gross=1.0)
    _, rp, ddp = run_portfolio(one, w, TRADE_START, COST)
    _, re, dde = run_engine(one[col], ema_ensemble(one[col]),
                            TRADE_START, COST)
    print("--- sanity: run_portfolio vs run_engine (1 instrument) ---")
    print(f"  run_engine    ret {re*100:+6.1f}%  dd {dde*100:6.1f}%")
    print(f"  run_portfolio ret {rp*100:+6.1f}%  dd {ddp*100:6.1f}%\n")


def score(name, weights, prices, spy, neutral=True):
    """Run one variant, return its row of stats + pass/fail flags."""
    res, ret, dd = run_portfolio(
        prices, weights, TRADE_START, COST,
        target_vol=0.10, vol_window=30, max_leverage=2.0)
    beta = portfolio_beta(prices, weights, spy, TRADE_START)
    by = yearly(res)
    lo_res, _, _ = run_portfolio(
        prices, dual_momentum_long_only(prices), TRADE_START, COST,
        target_vol=0.10)
    by_lo = yearly(lo_res)
    sp22 = by.get(2022, (0,))[0] - by_lo.get(2022, (0,))[0]

    ok_beta = abs(beta) < 0.25
    ok_sp = sp22 > 0
    ok_dd = dd > -0.40
    return {
        "name": name, "ret": ret, "dd": dd, "beta": beta,
        "sp22": sp22, "by": by,
        "ok_beta": ok_beta, "ok_sp": ok_sp, "ok_dd": ok_dd,
        "neutral": neutral,
    }


def main():
    prices = load_basket(BASKET, LOAD_START, END)
    print(f"Loaded {prices.shape[1]} instruments, "
          f"{prices.shape[0]} rows\n")
    spy = load_data("SPY", LOAD_START, END)

    sanity_check(prices)

    variants = {
        "base (symmetric MN)": (dual_momentum(prices), True),
        "1: tilt+dyn-gross": (
            dual_momentum_tilt(prices, benchmark=spy), False),
        "2: regime short": (
            dual_momentum_regime(prices, benchmark=spy), False),
        "3: vol-scaled MN": (dual_momentum_volscaled(prices), True),
        "4: cross_section_dual": (cross_sectional_dual_momentum(prices), True)
    }

    rows = [score(n, w, prices, spy, neutral=neu)
            for n, (w, neu) in variants.items()]

    print("=" * 72)
    print("  Variant comparison (target_vol=0.10)")
    print("=" * 72)
    print(f"{'Variant':<22}{'TotRet':>9}{'MaxDD':>9}{'Beta':>8}"
          f"{'Sp2022':>9}{'Tests':>8}")
    print("-" * 72)
    for r in rows:
        npass = sum([r["ok_beta"] or not r["neutral"],
                     r["ok_sp"], r["ok_dd"]])
        beta_note = "" if r["neutral"] else "*"
        print(f"{r['name']:<22}{r['ret']*100:>8.1f}%{r['dd']*100:>8.1f}%"
              f"{r['beta']:>+8.2f}{beta_note}{r['sp22']*100:>8.1f}%"
              f"{npass:>6}/3")
    print("-" * 72)
    print("  * net-long variants: beta~0 not expected (hedge, not MN)")
    print("  Sp2022 = market-neutral 2022 minus long-only 2022 (spread)")

    print("\n" + "=" * 72)
    print("  Spread by year (variant return minus long-only return)")
    print("=" * 72)
    lo_res, _, _ = run_portfolio(
        prices, dual_momentum_long_only(prices), TRADE_START, COST,
        target_vol=0.10)
    by_lo = yearly(lo_res)
    years = sorted(by_lo)
    hdr = "Year  " + "".join(f"{r['name'][:10]:>12}" for r in rows)
    print(hdr)
    for y in years:
        line = f"{y}  "
        for r in rows:
            sp = r["by"].get(y, (0,))[0] - by_lo.get(y, (0,))[0]
            line += f"{sp*100:>11.1f}%"
        tag = "  <- bear" if y == 2022 else ""
        print(line + tag)

    # --- Per-instrument yearly report for Cross Section Dual Momentum ---
    print("\n" + "=" * 72)
    print("  Per-instrument / per-year report: variant 4 (cross_section_dual)")
    print("=" * 72)
    w_cs = cross_sectional_dual_momentum(prices) # <-- Вызываем нашу стратегию
    run_portfolio_yearly(prices, w_cs, PERIODS, COST)


if __name__ == "__main__":
    main()