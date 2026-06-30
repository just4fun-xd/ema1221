"""Commodity cross-sectional viability diagnostic (Databento csv).

NOT a short backtest. With only 7 instruments a cross-sectional long/short
collapses to "1 long + 1 short" -- the same too-few-names failure that
closed the carry track. Instead this asks the NEW question equities never
answered: does commodities have the CORRELATION STRUCTURE that makes a
cross-section meaningful at all?

Three checks on the roll-adjusted Databento basket (CL NG GC SI HG ZW ZC):
  1. Pairwise return correlations -> low = real dispersion, high = dead.
  2. Momentum rank dispersion -> do instruments actually separate in
     a 126-day momentum ranking, or clump together?
  3. long-only momentum per instrument (sanity: are there trends at all).

Reads the M1 (front-month) series only: symbols like 'CL.c.0'.
Run locally: python -m diagnostics.commodity_diagnostic
"""

import pandas as pd
import numpy as np

from core.engine import run_engine
from strategies.strategies import ema_ensemble


CSV_PATH = "data/futures_m1_m2_fixed.csv"
FRONT_SUFFIX = ".c.0"   # M1 front-month continuous
COST = 0.001
TRADE_START = "2016-01-01"   # leave a year for the 200/256-day windows


def load_commodity_basket(csv_path=CSV_PATH, suffix=FRONT_SUFFIX):
    """Pivot the long-format Databento csv into a wide close-price frame.

    Args:
        csv_path: Path to the Databento export.
        suffix: Which continuous series to keep (front month .c.0).

    Returns:
        prices: DataFrame indexed by date, one column per instrument
            (CL, NG, GC, SI, HG, ZW, ZC), close prices.
    """
    df = pd.read_csv(csv_path, parse_dates=["ts_event"])
    df = df[df["symbol"].str.endswith(suffix)].copy()
    df["root"] = df["symbol"].str.replace(suffix, "", regex=False)
    df["date"] = df["ts_event"].dt.tz_localize(None).dt.normalize()
    prices = df.pivot_table(index="date", columns="root",
                            values="close", aggfunc="last").sort_index()
    return prices.ffill()


def main():
    prices = load_commodity_basket()
    print(f"Loaded {prices.shape[1]} instruments: "
          f"{list(prices.columns)}")
    print(f"Rows: {prices.shape[0]}, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}\n")

    rets = prices.pct_change()

    # --- CHECK 1: pairwise correlation ---
    print("=" * 60)
    print("CHECK 1: pairwise return correlation")
    print("=" * 60)
    corr = rets.corr()
    print(corr.round(2).to_string())
    # average off-diagonal correlation
    n = corr.shape[0]
    off = (corr.values.sum() - n) / (n * n - n)
    print(f"\n  avg off-diagonal correlation: {off:.2f}")
    print("  (equities mega-caps ~0.5-0.7; commodities lower = good)")
    if off < 0.3:
        print("  -> LOW correlation: real cross-sectional dispersion exists")
    else:
        print("  -> HIGH correlation: instruments move together, weak case")

    # --- CHECK 2: momentum rank dispersion ---
    print("\n" + "=" * 60)
    print("CHECK 2: momentum dispersion (126-day)")
    print("=" * 60)
    mom = prices.shift(21) / prices.shift(126) - 1
    mom = mom[mom.index >= TRADE_START]
    spread = (mom.max(axis=1) - mom.min(axis=1)).mean()
    print(f"  avg spread between best and worst momentum: {spread*100:.1f}%")
    print("  (wider spread = ranking separates instruments = good for L/S)")

    # --- CHECK 3: long-only momentum per instrument ---
    print("\n" + "=" * 60)
    print("CHECK 3: long-only momentum per instrument (trends exist?)")
    print("=" * 60)
    print(f"{'Instrument':<12}{'Return':>10}{'Max DD':>10}")
    print("-" * 32)
    wins = 0
    for col in prices.columns:
        pos = ema_ensemble(prices[col])
        _, ret, dd = run_engine(prices[col], pos, TRADE_START, COST)
        if ret > 0:
            wins += 1
        print(f"{col:<12}{ret*100:>9.1f}%{dd*100:>9.1f}%")
    print("-" * 32)
    print(f"  profitable: {wins}/{prices.shape[1]}")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print("  Low correlation + wide momentum spread => commodities HAS the")
    print("  structure equities lacked. Then a wide futures basket (20-30")
    print("  instruments via Databento) is worth buying for a real L/S test.")
    print("  High correlation OR narrow spread => same dead end as equities;")
    print("  close the commodity short idea cheaply, here, now.")
    print("\n  NOTE: 7 instruments is too few for a real L/S backtest. This")
    print("  diagnostic only decides whether buying MORE data is justified.")


if __name__ == "__main__":
    main()
