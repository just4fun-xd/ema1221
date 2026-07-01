"""P&L of Donchian champion decomposed by HMM regime -- on real CL.

The pivot we agreed on: instead of tuning HMM in a vacuum (2 vs 3 regimes,
ER window), let P&L decide how many regimes matter. Fit HMM, label each
day's regime by FORWARD FILTERING (no look-ahead), then measure the
Donchian champion's daily return SEPARATELY within each regime.

Readout decides the router design:
  - Donchian P&L very different in crisis vs calm, but SAME in trend vs
    range  -> 2 regimes suffice (crisis / calm). Don't fix trend/range.
  - Donchian P&L clearly different in trend vs range too -> 3 regimes
    carry info; improving trend/range separation is then worthwhile.
  - Donchian P&L flat across all regimes -> HMM carries no info for
    Donchian; a regime router won't help it. Close that idea.

Look-ahead discipline: regime_t uses data up to t only (predict_proba is
the forward filter). The strategy position already has run_engine's
shift(1). We align regime_t to the return EARNED on day t (which the
shifted position produced), so no future leaks into the attribution.

This is PER-INSTRUMENT by design (start with CL). No universal claim --
each instrument may answer differently, which is the whole point.

Run:  python -m diagnostics.pnl_by_regime
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from core.engine import run_engine
from core.load_panels import load_panels
from strategies.strategies_donchian_champions import (
    donchian_ensemble_macd_4step_take)

# reuse the v2 feature builder / labelling
from diagnostics.hmm_regime_v2 import (
    make_features, fit_hmm, label_states, N_STATES)


TRADE_START = "2020-07-01"
COST = 0.001


def regime_path(px):
    """Forward-filtered most-likely regime label per day (no look-ahead).

    Fits HMM on the full available history of THIS instrument, then uses
    predict_proba (forward filter) so regime_t depends only on data <= t.
    Returns a Series of string labels aligned to the feature index.
    """
    feat = make_features(px)
    model = fit_hmm(feat)
    labels = label_states(model)
    post = model.predict_proba(feat.values)      # forward posteriors
    state_seq = post.argmax(axis=1)
    return pd.Series([labels[s] for s in state_seq], index=feat.index), labels


def main():
    P = load_panels()
    for SYM in ["CL"]:                    # per-instrument; start with CL
        px = P["close"][SYM].where(P["native"][SYM]).dropna()
        print("=" * 64)
        print(f"  {SYM}: Donchian P&L by HMM regime  ({len(px)} days)")
        print("=" * 64)

        # 1. Donchian daily strategy returns (engine handles shift(1))
        pos = donchian_ensemble_macd_4step_take(px)
        res, tot, dd = run_engine(px, pos, TRADE_START, COST)
        daily = res["strategy"]           # per-day strategy return

        # 2. Regime label per day (forward-filtered)
        regimes, labels = regime_path(px)
        regimes = regimes.reindex(daily.index).ffill()

        print(f"  Overall: ret {tot*100:+.1f}%  dd {dd*100:.1f}%\n")

        # 3. Decompose: stats of Donchian daily return within each regime
        print(f"  {'regime':<9}{'days':>7}{'time%':>8}{'ann.ret':>10}"
              f"{'ann.vol':>10}{'sharpe':>8}{'hitrate':>9}")
        print("  " + "-" * 59)
        order = ["trend", "range", "crisis"]
        grand = {}
        for reg in order:
            d = daily[regimes == reg].dropna()
            if len(d) == 0:
                print(f"  {reg:<9}{'0':>7}")
                continue
            ann_ret = d.mean() * 252
            ann_vol = d.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
            hit = (d > 0).mean()
            share = len(d) / len(daily.dropna())
            grand[reg] = (ann_ret, sharpe)
            print(f"  {reg:<9}{len(d):>7}{share*100:>7.1f}%"
                  f"{ann_ret*100:>9.1f}%{ann_vol*100:>9.1f}%"
                  f"{sharpe:>8.2f}{hit*100:>8.1f}%")

        # 4. Readout: which split carries the P&L signal?
        print("\n  Interpretation:")
        if "crisis" in grand and ("trend" in grand or "range" in grand):
            calm = [grand[r][0] for r in ("trend", "range") if r in grand]
            calm_ret = np.mean(calm) if calm else float("nan")
            cr = grand["crisis"][0]
            print(f"    crisis ann.ret {cr*100:+.1f}%  vs  "
                  f"calm ann.ret {calm_ret*100:+.1f}%")
        if "trend" in grand and "range" in grand:
            tr, rg = grand["trend"][0], grand["range"][0]
            gap = (tr - rg) * 100
            print(f"    trend {tr*100:+.1f}%  vs  range {rg*100:+.1f}%  "
                  f"(gap {gap:+.1f}pp)")
            print("    -> big gap = 3 regimes carry info; small gap = 2")
            print("       regimes (crisis/calm) are enough for Donchian.")
        print("\n  NOTE: single fit on full history. If a regime's Donchian")
        print("  return flips sign vs the overall, that regime is where a")
        print("  filter could add value. Flat across regimes = HMM useless")
        print("  for Donchian on this instrument.")


if __name__ == "__main__":
    main()
