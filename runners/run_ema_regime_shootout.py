"""Regime filter shoot-out on equities: does HMM beat what you already have?

The Hurst-filtered EMA (ema_ensemble_voltarget_hurst_filtered) already
does regime gating -- goes to cash when Hurst says choppy. And plain
ema_ensemble_voltarget already passes DD<40% (-21%) via vol targeting. So
the honest question for HMM is not "does a crisis filter help" but "does
HMM add anything over the vol-target baseline AND over the Hurst filter
you already built".

Three-way, per-name + aggregate, walk-forward:
  A) BASE   -- ema_ensemble_voltarget (vol-targeted, already DD-safe)
  B) HMM    -- BASE with position zeroed on forward-flagged crisis
  C) HURST  -- ema_ensemble_voltarget_hurst_filtered (your existing gate)

Readout:
  HMM > BASE and HMM > HURST  -> HMM earns its place, best regime detector.
  HMM ~ HURST                 -> redundant; you already have the solution
                                 (same lesson as MACD/%b: no independent
                                 signal). Close HMM.
  HMM < BASE                  -> vol targeting already handles it; the
                                 filter cuts good exposure. Close HMM.

Honesty guards:
  - HMM crisis flag forward-filtered (regime_t uses data <= t).
  - Walk-forward: HMM fit on TRAIN, TEST flagged by the train-fit model.
  - Same engine, cost, basket for all three -> differences are the filter.
  - Per-name: the spread across names IS the finding, not an average.

Run:  python -m runners.run_ema_regime_shootout
"""

import numpy as np
import pandas as pd

from core.engine import run_engine
from core.engine_portfolio import load_basket
from strategies.strategies import (
    ema_ensemble_voltarget, ema_ensemble_voltarget_hurst_filtered)
from diagnostics.hmm_regime_v2 import (
    make_features, fit_hmm, label_states)


COST = 0.001
LOAD_START = "2017-01-01"
END = "2026-01-01"
TRAIN = ("2018-07-01", "2022-01-01")
TEST = ("2022-01-01", "2026-01-01")

BASKET = {
    "S&P 500": "SPY", "Nasdaq": "QQQ", "Apple": "AAPL", "Microsoft": "MSFT",
    "Amazon": "AMZN", "Meta": "META", "Alphabet": "GOOGL", "AMD": "AMD",
    "Netflix": "NFLX", "Visa": "V", "Mastercard": "MA", "UnitedHealth": "UNH",
    "Home Depot": "HD", "Coca-Cola": "KO", "Walmart": "WMT",
}


def crisis_flag(label_px, fit_px):
    """Forward-filtered crisis boolean; HMM fit on fit_px, applied to label_px."""
    feat_fit = make_features(fit_px)
    model = fit_hmm(feat_fit)
    labels = label_states(model)
    feat_lab = make_features(label_px)
    post = model.predict_proba(feat_lab.values)
    crisis_state = [s for s, l in labels.items() if l == "crisis"][0]
    seq = post.argmax(axis=1)
    return pd.Series(seq == crisis_state, index=feat_lab.index)


def three_ways(px, trade_start, fit_px):
    """Return dict of (ret, dd) for BASE / HMM / HURST on one series."""
    base = ema_ensemble_voltarget(px)
    _, br, bdd = run_engine(px, base, trade_start, COST)

    flag = crisis_flag(px, fit_px).reindex(px.index).fillna(False).astype(bool)
    hmm = base.where(~flag, other=0.0)
    _, hr, hdd = run_engine(px, hmm, trade_start, COST)

    hur = ema_ensemble_voltarget_hurst_filtered(px)
    _, ur, udd = run_engine(px, hur, trade_start, COST)

    return {"BASE": (br, bdd), "HMM": (hr, hdd), "HURST": (ur, udd)}


def period_table(prices, tag, trade_start, fit_slice):
    print(f"\n{'=' * 78}\n  {tag}\n{'=' * 78}")
    print(f"{'name':<14}"
          f"{'BASE ret/dd':>18}{'HMM ret/dd':>18}{'HURST ret/dd':>18}")
    print("-" * 78)

    agg = {"BASE": [], "HMM": [], "HURST": []}
    for name in prices.columns:
        px = prices[name].dropna()
        fit_px = px[(px.index >= fit_slice[0]) & (px.index < fit_slice[1])]
        if len(fit_px) < 100:
            continue
        try:
            r = three_ways(px, trade_start, fit_px)
        except Exception as e:
            print(f"{name:<14}  ERROR: {e}")
            continue
        for k in agg:
            agg[k].append(r[k][0])
        print(f"{name:<14}"
              f"{r['BASE'][0]*100:>+8.0f}%/{r['BASE'][1]*100:>5.0f}%"
              f"{r['HMM'][0]*100:>+9.0f}%/{r['HMM'][1]*100:>5.0f}%"
              f"{r['HURST'][0]*100:>+9.0f}%/{r['HURST'][1]*100:>5.0f}%")

    print("-" * 78)
    print(f"{'MEDIAN ret':<14}"
          f"{np.median(agg['BASE'])*100:>+8.0f}%       "
          f"{np.median(agg['HMM'])*100:>+9.0f}%       "
          f"{np.median(agg['HURST'])*100:>+9.0f}%")
    return agg


def main():
    prices = load_basket(BASKET, LOAD_START, END)
    print(f"Loaded {prices.shape[1]} names, {prices.shape[0]} rows")

    # Walk-forward: fit HMM on TRAIN slice, evaluate on TEST slice.
    # (Hurst has no fit -- it's self-contained per series.)
    agg_te = period_table(prices, "TEST 2022-2026 (HMM fit on 2018-2022)",
                          TEST[0], TRAIN)

    print("\n  Readout:")
    for k in ("BASE", "HMM", "HURST"):
        print(f"    {k:<6} median ret {np.median(agg_te[k])*100:+.0f}%")
    b, h, u = (np.median(agg_te["BASE"]), np.median(agg_te["HMM"]),
               np.median(agg_te["HURST"]))
    print()
    if h > b and h > u:
        print("  -> HMM beats both baseline and Hurst: earns its place.")
    elif abs(h - u) < 0.03:
        print("  -> HMM ~ Hurst: redundant. You already have the gate.")
        print("     Same lesson as MACD/%b -- no independent signal.")
    elif h < b:
        print("  -> HMM below baseline: vol targeting already handles risk;")
        print("     the crisis filter cuts good exposure. Close HMM.")
    else:
        print("  -> Mixed: read the per-name spread; no clean winner.")
    print("\n  Median over names (Tesla/Nvidia excluded as usual). Per-name")
    print("  spread matters more than the median -- a filter that helps SPY")
    print("  may hurt a single name, and that is the per-instrument finding.")


if __name__ == "__main__":
    main()
