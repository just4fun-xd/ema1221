"""Wide-basket test: dual_momentum on the full S&P 500.

Two questions, two variants compared side by side:
  - tilt (net-long, asymmetric legs): does concentration dissolve on a
    wide universe? (it did: top-3 P&L share 66% -> ~6%)
  - base (symmetric market-neutral): with beta ~0, is there REAL alpha
    from the long-minus-short spread, or was the wide-basket return just
    market exposure (tilt's beta was +0.60)?

The market-neutral return is the decisive test: if positive at beta~0,
the short leg carries genuine cross-sectional alpha on a wide basket.
If ~0, the method is a risk tool, not an alpha source.

SURVIVORSHIP BIAS: current members only -> absolute returns overstated.
The market-neutral SIGN and the concentration RATIO are the robust
signals here, not the absolute level.
"""

from core.engine import load_data
from core.engine_portfolio import run_portfolio, portfolio_beta
from strategies.strategies_dualmom import dual_momentum, dual_momentum_tilt
from core.universe import get_sp500_tickers, load_basket_batch

LOAD_START = "2017-01-01"
END = "2026-01-01"
TRADE_START = "2018-07-01"
COST = 0.001

# Set LIMIT=None for the full 500; a smaller number runs faster while testing.
LIMIT = None


def concentration(prices, weights):
    """Return (top1, top3, top10) P&L shares and avg active names."""
    held = weights.shift(1)
    mask = held.index >= TRADE_START
    active = (held.abs() > 1e-9).sum(axis=1)[mask].mean()
    contrib = (prices.pct_change() * held)[mask].sum()
    gross = contrib.abs().sum()
    ranked = contrib.abs().sort_values(ascending=False)
    t1 = ranked.iloc[0] / gross * 100
    t3 = ranked.iloc[:3].sum() / gross * 100
    t10 = ranked.iloc[:10].sum() / gross * 100
    return t1, t3, t10, active, ranked


def main():
    tickers = get_sp500_tickers(limit=LIMIT)
    print(f"Universe: {len(tickers)} tickers (current S&P 500)")
    print("NOTE: survivorship-biased (current members only).\n")

    prices = load_basket_batch(tickers, LOAD_START, END)
    print(f"Loaded {prices.shape[1]} usable instruments, "
          f"{prices.shape[0]} rows\n")

    spy = load_data("SPY", LOAD_START, END)

    variants = {
        "base (market-neutral)": dual_momentum(prices),
        "tilt (net-long)": dual_momentum_tilt(prices, benchmark=spy),
    }

    print("=" * 64)
    print("  WIDE BASKET: market-neutral vs tilt on S&P 500")
    print("=" * 64)
    print(f"{'Variant':<24}{'TotRet':>9}{'MaxDD':>9}{'Beta':>8}"
          f"{'top3 P&L':>10}")
    print("-" * 64)

    results = {}
    for name, w in variants.items():
        _, ret, dd = run_portfolio(prices, w, TRADE_START, COST,
                                   target_vol=0.10)
        beta = portfolio_beta(prices, w, spy, TRADE_START)
        t1, t3, t10, active, ranked = concentration(prices, w)
        results[name] = (ret, dd, beta, t3, active)
        print(f"{name:<24}{ret*100:>8.1f}%{dd*100:>8.1f}%"
              f"{beta:>+8.2f}{t3:>9.1f}%")

    print("-" * 64)
    print("  (narrow 19-name basket had top-3 P&L share = 66%)")

    ret_mn, dd_mn, beta_mn, t3_mn, act_mn = results["base (market-neutral)"]
    print()
    print("=" * 64)
    print("  DECISIVE TEST: is there alpha at beta ~ 0?")
    print("=" * 64)
    print(f"  market-neutral return : {ret_mn*100:+.1f}%")
    print(f"  market-neutral beta   : {beta_mn:+.2f}")
    print(f"  market-neutral max DD : {dd_mn*100:.1f}%")
    print(f"  avg active names      : {act_mn:.0f} of {prices.shape[1]}")
    print()
    if abs(beta_mn) < 0.25 and ret_mn > 0.10:
        print("  -> POSITIVE return at near-zero beta: the short leg")
        print("     carries genuine cross-sectional alpha on a wide basket.")
    elif abs(beta_mn) < 0.25:
        print("  -> ~zero return at near-zero beta: NO alpha. The method is")
        print("     a risk tool, not an alpha source. tilt's return was beta.")
    else:
        print("  -> beta not near zero; legs not balanced on this basket.")
    print()
    print("  REMINDER: survivorship-biased universe -> real numbers lower.")


if __name__ == "__main__":
    main()