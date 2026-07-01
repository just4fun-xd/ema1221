"""
Preprocess futures_basket_wide.csv -> aligned panels for cross-sectional momentum.

Handles the calendar mismatch (energy/metals ~1553 days vs grains ~1213) via
union-reindex + ffill, WITH the three guards we agreed on:

  1. carry (roll_yield) is computed on NATIVE dates only (both legs present),
     THEN ffilled. Never ffill M1/M2 independently and divide — that fabricates
     roll-yield jumps on alignment artifacts.
  2. volume is set to 0 on ffilled days (not carried) so the liquidity filter
     doesn't think a closed market traded.
  3. an is_native mask marks days each instrument actually traded. The
     cross-sectional ranker MUST restrict each date's cross-section to
     is_native==True instruments — never rank/enter on a ffilled (closed) day.

Dropped from roster: PA (med_vol 1495), ZO (207), ZR (286) — too thin.

Output: one parquet with a MultiIndex (date) x columns, plus the mask.
Panels produced: close, volume, roll_yield, is_native  (17 instruments each).
"""

import pandas as pd

SRC = "/Users/shalygin/dev/Python_work/EMA1221/data/futures_basket_wide.csv"
OUT_DIR = "/Users/shalygin/dev/Python_work/EMA1221/data/"

DROP = {"PA", "ZO", "ZR"}

# Roll type must match how it was downloaded (metals = volume roll .v)
VOLUME_ROLL = {"GC", "SI", "HG", "PL", "PA"}

ROOTS_ALL = [
    "CL", "NG", "HO", "RB",
    "GC", "SI", "HG", "PL", "PA",
    "ZW", "KE", "ZC", "ZS", "ZM", "ZL", "ZO",
    "LE", "GF", "HE",
    "ZR",
]
ROOTS = [r for r in ROOTS_ALL if r not in DROP]  # 17


def sym(root, n):
    stype = "v" if root in VOLUME_ROLL else "c"
    return f"{root}.{stype}.{n}"


def main():
    raw = pd.read_csv(SRC, index_col=0, parse_dates=True)
    # raw index is the bar timestamp; ensure it's a clean daily DatetimeIndex
    raw.index = pd.to_datetime(raw.index).normalize()

    # Split each instrument into M1 / M2 close + M1 volume
    m1_close, m2_close, m1_vol = {}, {}, {}
    for r in ROOTS:
        s1 = raw[raw["symbol"] == sym(r, 0)]
        s2 = raw[raw["symbol"] == sym(r, 1)]
        if s1.empty:
            print(f"  WARN: {r} front month missing, skipping")
            continue
        m1_close[r] = s1["close"].groupby(level=0).last()
        m1_vol[r]   = s1["volume"].groupby(level=0).last()
        if not s2.empty:
            m2_close[r] = s2["close"].groupby(level=0).last()

    close_native = pd.DataFrame(m1_close)   # NaN where instrument didn't trade
    vol_native   = pd.DataFrame(m1_vol)
    m2_native    = pd.DataFrame(m2_close)

    # --- native-day mask BEFORE any ffill --------------------------------
    is_native = close_native.notna()

    # --- union calendar --------------------------------------------------
    cal = close_native.index.union(vol_native.index)
    close   = close_native.reindex(cal)
    vol     = vol_native.reindex(cal)
    m2      = m2_native.reindex(cal)
    native  = is_native.reindex(cal).fillna(False)

    # --- carry on native dates ONLY, then ffill --------------------------
    # roll_yield_t = (M1 - M2) / M2, valid only where BOTH legs are native.
    both_native = close_native.reindex(cal).notna() & m2.notna()
    roll_yield = (close - m2) / m2
    roll_yield = roll_yield.where(both_native)   # blank alignment artifacts
    roll_yield = roll_yield.ffill()              # carry persists between rolls

    # --- ffill prices; zero volume on ffilled days -----------------------
    close = close.ffill()
    # volume: real value on native days, 0 on ffilled (closed) days
    vol = vol.where(native, other=0.0)

    # leading NaNs (before an instrument's first trade) stay NaN — correct,
    # the ranker's is_native mask already excludes them.

    # --- sanity -----------------------------------------------------------
    print(f"Instruments: {close.shape[1]}  Calendar days: {close.shape[0]}")
    print(f"{'sym':<6}{'native':>8}{'ffilled':>9}{'first_native':>14}")
    for r in close.columns:
        n_native = int(native[r].sum())
        n_ff = int(close.shape[0] - n_native)
        first = native[r].idxmax() if native[r].any() else None
        print(f"{r:<6}{n_native:>8}{n_ff:>9}{str(first.date()) if first is not None else 'NA':>14}")

    # --- save -------------------------------------------------------------
    close.to_parquet(OUT_DIR + "panel_close.parquet")
    vol.to_parquet(OUT_DIR + "panel_volume.parquet")
    roll_yield.to_parquet(OUT_DIR + "panel_rollyield.parquet")
    native.to_parquet(OUT_DIR + "panel_native.parquet")
    print(f"\nSaved 4 panels -> {OUT_DIR}panel_*.parquet")


if __name__ == "__main__":
    main()
