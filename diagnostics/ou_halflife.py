"""OU diagnostic: which of the 17 futures are mean-reverting enough to trade.

Trek 2.1 step 0. Before writing an OU strategy, MEASURE mean-reversion
per instrument -- do not assume it from the roadmap. DM failed on
commodities precisely BECAUSE they mean-revert; this quantifies how fast,
per name, so OU is applied only where it can work.

WHAT IS MEASURED
  half-life h = ln(2)/theta, where theta is the OU mean-reversion speed
  estimated by one OLS regression:
        dX_t = a + b * X_{t-1} + eps      (discrete OU / AR(1))
        theta = -b,   h = ln(2)/theta     (in trading days)
  Small h (< ~20d)  -> deviation decays fast -> OU tradeable.
  Large h (> ~60d)  -> too slow -> OU holds against the move for months.
  b >= 0            -> NOT mean-reverting (trending / random walk).

WHAT SERIES
  OU trades the DEVIATION from a local mean, not the raw price. Raw
  commodity prices trend / random-walk (half-life in the hundreds).
  So X = close - SMA(window): the detrended deviation the strategy would
  actually trade. Half-life is reported on THAT, which is the honest
  question ("does the deviation revert?"), not "does the price revert?".

  Reported for two SMA windows so the answer isn't an artifact of one
  detrending choice.

ADF test: augmented Dickey-Fuller on the deviation. p < 0.05 -> reject
unit root -> deviation is stationary (mean-reverting) with statistical
confidence, not just a point estimate of half-life.

Run:  python -m diagnostics.ou_halflife
"""

import numpy as np
import pandas as pd

from core.load_panels import load_panels

try:
    from statsmodels.tsa.stattools import adfuller
    HAVE_ADF = True
except Exception:
    HAVE_ADF = False


def half_life(deviation):
    """Estimate OU half-life (days) from a deviation series via OLS AR(1).

    Args:
        deviation: Series of the detrended series X (close - SMA).

    Returns:
        (half_life_days, theta, b). half_life is np.inf if b >= 0
        (no mean reversion). NaNs dropped internally.
    """
    x = deviation.dropna()
    x_lag = x.shift(1).dropna()
    x = x.loc[x_lag.index]
    if len(x) < 60:
        return np.nan, np.nan, np.nan

    dx = x - x_lag
    # OLS: dx = a + b * x_lag
    X = np.column_stack([np.ones(len(x_lag)), x_lag.values])
    beta, *_ = np.linalg.lstsq(X, dx.values, rcond=None)
    a, b = beta[0], beta[1]

    theta = -b
    if theta <= 0:
        return np.inf, theta, b   # not mean-reverting
    hl = np.log(2) / theta
    return hl, theta, b


def adf_p(deviation):
    """ADF p-value on the deviation (unit-root test). Lower = more stationary."""
    if not HAVE_ADF:
        return np.nan
    x = deviation.dropna()
    if len(x) < 60:
        return np.nan
    try:
        return adfuller(x, autolag="AIC")[1]
    except Exception:
        return np.nan


def main():
    P = load_panels()
    close = P["close"]
    native = P["native"]
    print(f"Loaded {close.shape[1]} instruments, {close.shape[0]} days\n")

    windows = [20, 60]   # SMA windows for detrending

    print(f"{'sym':<5}", end="")
    for w in windows:
        print(f"{'hl@'+str(w):>9}{'adf@'+str(w):>9}", end="")
    print(f"{'verdict':>14}")
    print("-" * (5 + len(windows) * 18 + 14))

    rows = []
    for sym in close.columns:
        # Use native days only for the fit (ffilled days are flat -> bias
        # theta toward zero / inflate half-life).
        px = close[sym].where(native[sym]).dropna()
        line = f"{sym:<5}"
        best_hl = np.inf
        for w in windows:
            sma = px.rolling(w).mean()
            dev = px - sma
            hl, theta, b = half_life(dev)
            p = adf_p(dev)
            best_hl = min(best_hl, hl if np.isfinite(hl) else np.inf)
            hl_str = f"{hl:>9.1f}" if np.isfinite(hl) else f"{'inf':>9}"
            p_str = f"{p:>9.3f}" if not np.isnan(p) else f"{'--':>9}"
            line += hl_str + p_str
        # Verdict on the fastest of the two windows
        if not np.isfinite(best_hl):
            verdict = "no-revert"
        elif best_hl < 20:
            verdict = "OU strong"
        elif best_hl < 60:
            verdict = "OU maybe"
        else:
            verdict = "too slow"
        line += f"{verdict:>14}"
        print(line)
        rows.append((sym, best_hl, verdict))

    print("\nGuide: hl = half-life in trading days (ln2/theta).")
    print("  OU strong (<20d) -> deviation reverts fast, good OU candidate")
    print("  OU maybe (20-60d) -> borderline, test with care")
    print("  too slow (>60d) / no-revert -> OU will hold against the move")
    if not HAVE_ADF:
        print("  [adf blank: statsmodels not importable in this run]")
    else:
        print("  adf p<0.05 -> deviation statistically stationary")

    # Shortlist
    strong = [s for s, h, v in rows if v == "OU strong"]
    maybe = [s for s, h, v in rows if v == "OU maybe"]
    print(f"\nOU candidates -- strong: {strong or 'none'}")
    print(f"OU candidates -- maybe:  {maybe or 'none'}")
    if not strong and not maybe:
        print("  -> No fast mean-reverters. OU may not fit this basket;")
        print("     that itself is a finding (report, don't force it).")


if __name__ == "__main__":
    main()
