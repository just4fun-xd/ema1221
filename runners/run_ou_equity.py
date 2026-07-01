"""OU on equities: does mean-reversion work better on stocks than commodities?

Hypothesis under test (Kirill): OU failed on commodities because
commodities don't revert -- maybe OU is really an equity strategy.

Prediction (to be confirmed or refuted): OU should do WORSE on equities,
because equities trend UP structurally (the drift that made EMA work and
the equity short leg fail). Shorting a structurally rising stock on "too
high" catches the drift, same failure as the equity short leg.

The one way the hypothesis could WIN: short-term reversal. Individual
stocks' daily/weekly moves partly revert (a documented effect), unlike
indices which trend. OU's ~7-9 day half-life sits in that zone. So this
runner splits the basket into two groups:

  NAMES   (AAPL, MSFT, ...) -- where short-term reversal may exist
  INDICES (SPY, QQQ)        -- CONTROL: reversal should NOT exist here

Clean result: OU fails on INDICES but works on NAMES -> short-term
reversal is real, OU belongs on single stocks. OU fails on both -> the
problem was never the asset class, it's regime dependence (OU shorts
trends and blows up); commodities aren't special.

Method mirrors run_ou_walkforward.py exactly (same shuffle control, same
TRAIN/TEST split, same warmup) so any difference is attributable to the
asset class, not the test.

Run:  python -m runners.run_ou_equity
"""

import numpy as np
import pandas as pd

from core.engine import run_spread_engine
from core.engine_portfolio import load_basket
from strategies.strategies_ou import ou_zscore


N_SHUFFLE = 30
COST = 0.001
WARMUP = 40

LOAD_START = "2017-01-01"
END = "2026-01-01"
TRAIN = ("2018-07-01", "2022-01-01")     # ~3.5y train
TEST = ("2022-01-01", "2026-01-01")      # 2022 bear + 2023-25 recovery

INDICES = {"S&P 500": "SPY", "Nasdaq": "QQQ"}
NAMES = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN", "Meta": "META",
    "Alphabet": "GOOGL", "AMD": "AMD", "Netflix": "NFLX", "Visa": "V",
    "Mastercard": "MA", "UnitedHealth": "UNH", "Home Depot": "HD",
    "Bank of America": "BAC", "Coca-Cola": "KO", "Procter & Gamble": "PG",
    "Walmart": "WMT",
}
# Tesla/Nvidia excluded on purpose: extreme trenders, would just confirm
# "OU shorts a rocket and dies" without informing the reversal question.


def _slice(s, start, end):
    return s[(s.index >= start) & (s.index < end)]


def real_return(px, trade_start, cost):
    pos = ou_zscore(px)
    _, ret, dd = run_spread_engine(px, pos, trade_start, cost)
    return ret, dd


def shuffle_returns(px, seed):
    rng = np.random.default_rng(seed)
    rets = px.pct_change().dropna().values
    shuffled = rng.permutation(rets)
    synth = np.empty(len(shuffled) + 1)
    synth[0] = px.iloc[0]
    synth[1:] = synth[0] * np.cumprod(1 + shuffled)
    return pd.Series(synth, index=px.index[:len(synth)])


def evaluate_period(px, start, end):
    px = _slice(px, start, end).dropna()
    if len(px) < WARMUP + 60:
        return np.nan, np.nan, np.nan, False
    trade_start = px.index[WARMUP]
    rr, dd = real_return(px, trade_start, COST)
    shuf = []
    for seed in range(N_SHUFFLE):
        s = shuffle_returns(px, seed)
        sr, _ = real_return(s, trade_start, COST)
        if np.isfinite(sr):
            shuf.append(sr)
    p90 = np.percentile(shuf, 90) if shuf else np.nan
    passed = np.isfinite(rr) and np.isfinite(p90) and rr > p90
    return rr, dd, p90, passed


def run_group(prices, label, group_cols):
    print(f"\n{'=' * 78}\n  {label}\n{'=' * 78}")
    print(f"{'sym':<14}"
          f"{'tr real':>9}{'tr p90':>9}{'tr?':>5}   "
          f"{'te real':>9}{'te p90':>9}{'te?':>5}   "
          f"{'te dd':>8}{'verdict':>10}")
    print("-" * 78)

    both = []
    for name in group_cols:
        if name not in prices.columns:
            continue
        px = prices[name].dropna()
        trr, trdd, trp90, trok = evaluate_period(px, *TRAIN)
        ter, tedd, tep90, teok = evaluate_period(px, *TEST)

        if trok and teok:
            verdict = "REAL"
            both.append(name)
        elif trok and not teok:
            verdict = "faded"
        elif teok:
            verdict = "test-only"
        else:
            verdict = "-"

        def f(v):
            return f"{v*100:>8.1f}%" if np.isfinite(v) else f"{'--':>9}"

        print(f"{name:<14}"
              f"{f(trr)}{f(trp90)}{('Y' if trok else 'n'):>5}   "
              f"{f(ter)}{f(tep90)}{('Y' if teok else 'n'):>5}   "
              f"{f(tedd)}{verdict:>10}")
    return both


def main():
    all_tickers = {**INDICES, **NAMES}
    prices = load_basket(all_tickers, LOAD_START, END)
    print(f"Loaded {prices.shape[1]} instruments, {prices.shape[0]} rows")
    print(f"TRAIN {TRAIN[0]}..{TRAIN[1]}   TEST {TEST[0]}..{TEST[1]}")
    print(f"Shuffle seeds {N_SHUFFLE}, cost {COST*1e4:.0f}bps")

    idx_real = run_group(prices, "CONTROL: INDICES (reversal NOT expected)",
                         list(INDICES))
    nm_real = run_group(prices, "NAMES (short-term reversal may exist)",
                        list(NAMES))

    print(f"\n{'=' * 78}\n  READOUT\n{'=' * 78}")
    print(f"  INDICES passing BOTH periods: {idx_real or 'none'}")
    print(f"  NAMES   passing BOTH periods: {nm_real or 'none'}")
    print()
    if nm_real and not idx_real:
        print("  -> NAMES work, INDICES don't: SHORT-TERM REVERSAL confirmed.")
        print("     OU belongs on single stocks, not indices/commodities.")
        print("     Kirill's hypothesis WINS -- develop OU on equity names.")
    elif not nm_real and not idx_real:
        print("  -> OU fails on BOTH equity groups too. The problem was never")
        print("     the asset class -- it's regime dependence (OU shorts")
        print("     trends and blows up). Commodities weren't special.")
        print("     Confirms: OU needs a regime detector (HMM), not a new")
        print("     asset class. Compare these numbers to the commodity run.")
    else:
        print("  -> Mixed/other pattern -- read the per-name table above;")
        print("     the clean reversal signature (names yes, indices no)")
        print("     did not appear.")


if __name__ == "__main__":
    main()
