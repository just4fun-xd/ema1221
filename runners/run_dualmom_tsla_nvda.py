"""Step 1 diagnostic: how much of the equity dual-momentum result is
Tesla + Nvidia, versus the ranking edge itself.

Runs each existing DM variant TWICE -- on the full 19-name basket and on
the same basket with TSLA and NVDA removed -- and prints the delta.
No new strategy code: calls the same functions run_dualmom.py uses, with
identical run_portfolio params, so the full-basket column reproduces the
main run exactly (built-in correctness check).

Read of the deltas:
  small delta  -> edge is real, not two meme names.
  large delta  -> result was largely a bet on TSLA/NVDA momentum.
The long-only row matters most: that is where a mega-cap winner rides
straight into P&L. Market-neutral should be more robust (short leg
partly offsets), and that contrast is itself the finding.

Run:  python -m runners.run_dualmom_tsla_nvda
"""

import pandas as pd

from core.engine import load_data
from strategies.strategies import cross_sectional_dual_momentum
from core.engine_portfolio import (load_basket, run_portfolio, portfolio_beta)
from strategies.strategies_dualmom import (
    dual_momentum, dual_momentum_long_only, dual_momentum_tilt,
    dual_momentum_regime, dual_momentum_volscaled)

from runners.run_dualmom import BASKET, LOAD_START, END, TRADE_START, COST


EXCLUDE = ["Tesla", "Nvidia"]   # keys in BASKET


def build_variants(prices, spy):
    """Same variant set as the main runner, bound to a given basket."""
    return {
        "base (symm MN)":     (dual_momentum(prices), True),
        "long-only":          (dual_momentum_long_only(prices), False),
        "1: tilt+dyn-gross":  (dual_momentum_tilt(prices, benchmark=spy), False),
        "2: regime short":    (dual_momentum_regime(prices, benchmark=spy), False),
        "3: vol-scaled MN":   (dual_momentum_volscaled(prices), True),
        "4: cross_section":   (cross_sectional_dual_momentum(prices), True),
    }


def run_one(prices, weights, spy):
    res, ret, dd = run_portfolio(
        prices, weights, TRADE_START, COST,
        target_vol=0.10, vol_window=30, max_leverage=2.0)
    beta = portfolio_beta(prices, weights, spy, TRADE_START)
    # 2022 return, isolated
    r2022 = None
    for y, g in res.groupby(res.index.year):
        if y == 2022:
            r2022 = g["equity"].iloc[-1] / g["equity"].iloc[0] - 1
    return ret, dd, beta, r2022


def main():
    spy = load_data("SPY", LOAD_START, END)

    full = load_basket(BASKET, LOAD_START, END)
    reduced_basket = {k: v for k, v in BASKET.items() if k not in EXCLUDE}
    reduced = load_basket(reduced_basket, LOAD_START, END)

    print(f"Full basket:    {full.shape[1]} names, {full.shape[0]} rows")
    print(f"Reduced basket: {reduced.shape[1]} names "
          f"(dropped {', '.join(EXCLUDE)})\n")

    vfull = build_variants(full, spy)
    vred = build_variants(reduced, spy)

    print("=" * 88)
    print("  TSLA/NVDA contribution: full basket vs basket minus two meme names")
    print("=" * 88)
    print(f"{'Variant':<20}"
          f"{'Ret full':>10}{'Ret -2':>10}{'dRet':>9}"
          f"{'DD full':>10}{'DD -2':>9}"
          f"{'Beta -2':>9}{'2022 -2':>9}")
    print("-" * 88)

    for name in vfull:
        rf, ddf, bf, _ = run_one(full, vfull[name][0], spy)
        rr, ddr, br, r22r = run_one(reduced, vred[name][0], spy)
        dret = rr - rf
        print(f"{name:<20}"
              f"{rf*100:>9.1f}%{rr*100:>9.1f}%{dret*100:>+8.1f}%"
              f"{ddf*100:>9.1f}%{ddr*100:>8.1f}%"
              f"{br:>+9.2f}{(r22r*100 if r22r is not None else 0):>8.1f}%")

    print("-" * 88)
    print("  dRet = reduced minus full. Large negative -> that variant's")
    print("  return leaned heavily on Tesla/Nvidia. long-only is the")
    print("  stress case; market-neutral rows should move less.")


if __name__ == "__main__":
    main()
