"""Commodity cross-sectional dual momentum -- validation vs Donchian baseline.

The commodity analogue of run_dualmom.py. Reads the preprocessed panels
(17 roll-adjusted futures, native mask, carry separate), runs the
market-neutral / long-only / vol-scaled DM variants, and compares against
the trend-following baseline on the SAME instruments -- because a lone DM
number is meaningless without "better/worse than the trend champion".

Key differences from the equity runner:
  - No SPY beta test (commodity basket has no single market proxy). The
    neutrality check is gross-net ~ 0 by construction; the honest test is
    DM vs Donchian on identical instruments.
  - Walk-forward: train 2020-2022 (parameter choice), test 2023-2024.
  - The 2020-2024 window is regime-specific (covid crash, 2021-22
    commodity bull, 2023-24 mean-reversion) -- results are labelled as
    such, not extrapolated to all regimes.

Run:  python -m runners.run_dualmom_commodity

Donchian baseline = your real champion (donchian_ensemble_macd_4step_take).
It is LONG-ONLY and carries its OWN vol-targeting internally, so it is run
through run_portfolio with target_vol=None (no double-scaling). The DM
variants return raw weights and ARE vol-targeted by the engine.
"""

import numpy as np
import pandas as pd

from core.engine_portfolio import run_portfolio, positions_to_weights
from core.load_panels import load_panels
from strategies.strategies_dualmom_commodity import (
    dual_momentum_commodity, dual_momentum_commodity_long_only)
from strategies.strategies_donchian_champions import (
    donchian_ensemble_macd_4step_take)
from strategies.strategies_donchian_champions import (
    donchian_ensemble_macd_4step_take)


COST = 0.001          # match project default (run_engine/run_portfolio) so
                      # DM and Donchian are compared on identical costs
TRADE_START = "2020-07-01"
TARGET_VOL = 0.12
VOL_WINDOW = 30
MAX_LEV = 2.0

# Walk-forward split
TRAIN = ("2020-01-01", "2023-01-01")
TEST = ("2023-01-01", "2025-01-01")


def stats(res, ret, dd, label):
    print(f"  {label:<26} ret {ret*100:>+7.1f}%   dd {dd*100:>6.1f}%")
    return {"label": label, "ret": ret, "dd": dd, "res": res}


def yearly(res):
    out = {}
    for y, g in res.groupby(res.index.year):
        out[y] = g["equity"].iloc[-1] / g["equity"].iloc[0] - 1
    return out


def run_window(close, native, start, tag):
    """Run the variant set on [start, end-of-data], print a block."""
    print(f"\n{'=' * 60}\n  {tag}  (trade start {start})\n{'=' * 60}")

    out = []

    # --- DM variants ---
    w_mn = dual_momentum_commodity(close, native=native)
    r = run_portfolio(close, w_mn, start, COST, target_vol=TARGET_VOL,
                      vol_window=VOL_WINDOW, max_leverage=MAX_LEV)
    out.append(stats(*r, "DM market-neutral"))

    w_lo = dual_momentum_commodity_long_only(close, native=native)
    r = run_portfolio(close, w_lo, start, COST, target_vol=TARGET_VOL,
                      vol_window=VOL_WINDOW, max_leverage=MAX_LEV)
    out.append(stats(*r, "DM long-only"))

    w_vs = dual_momentum_commodity(close, native=native, vol_window=60)
    r = run_portfolio(close, w_vs, start, COST, target_vol=TARGET_VOL,
                      vol_window=VOL_WINDOW, max_leverage=MAX_LEV)
    out.append(stats(*r, "DM vol-scaled MN"))

    # --- Donchian champion on the SAME instruments ---
    # LONG-ONLY and self-vol-targeted -> run with target_vol=None to avoid
    # double-scaling. This is the trend baseline DM must beat.
    w_don = positions_to_weights(
        close, donchian_ensemble_macd_4step_take, gross=1.0)
    r = run_portfolio(close, w_don, start, COST, target_vol=None)
    out.append(stats(*r, "Donchian champion (LO)"))

    return out


def main():
    P = load_panels()
    close, native = P["close"], P["native"]
    print(f"Loaded panels: {close.shape[1]} instruments, {close.shape[0]} days")
    print(f"Instruments: {', '.join(close.columns)}")

    # Carry context (last obs) -- interpretation aid, not a signal here.
    ry = P["rollyield"].iloc[-1].sort_values()
    print("\nRoll-yield (last obs, contango<0<backwardation):")
    print("  " + "  ".join(f"{k}:{v:+.3f}" for k, v in ry.items()))

    # --- Full window ---
    full = run_window(close, native, TRADE_START, "FULL 2020-2024")

    # --- Walk-forward: train then test ---
    close_tr = close.loc[:TRAIN[1]]
    native_tr = native.loc[:TRAIN[1]]
    run_window(close_tr, native_tr, TRAIN[0], "TRAIN 2020-2022")

    close_te = close.loc[TEST[0]:]
    native_te = native.loc[TEST[0]:]
    run_window(close_te, native_te, TEST[0], "TEST 2023-2024")

    # --- Yearly spread: MN minus long-only (short-leg value) ---
    print(f"\n{'=' * 60}\n  Short-leg value: MN minus long-only, by year\n{'=' * 60}")
    w_mn = dual_momentum_commodity(close, native=native)
    w_lo = dual_momentum_commodity_long_only(close, native=native)
    res_mn = run_portfolio(close, w_mn, TRADE_START, COST,
                           target_vol=TARGET_VOL, vol_window=VOL_WINDOW,
                           max_leverage=MAX_LEV)[0]
    res_lo = run_portfolio(close, w_lo, TRADE_START, COST,
                           target_vol=TARGET_VOL, vol_window=VOL_WINDOW,
                           max_leverage=MAX_LEV)[0]
    ymn, ylo = yearly(res_mn), yearly(res_lo)
    for y in sorted(ymn):
        sp = ymn[y] - ylo.get(y, 0.0)
        print(f"  {y}   MN {ymn[y]*100:>+6.1f}%   LO {ylo.get(y,0)*100:>+6.1f}%"
              f"   spread {sp*100:>+6.1f}%")

    print("\n  Donchian champion is LONG-ONLY + self-vol-targeted (run with")
    print("  target_vol=None). Compare it against DM long-only to isolate")
    print("  ranking value from directional exposure.")
    print("  Window 2020-2024 is regime-specific; do not extrapolate.")


if __name__ == "__main__":
    main()
