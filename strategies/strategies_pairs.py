import numpy as np
import pandas as pd

from core.engine import load_data, run_engine


def zscore_spread(close_a, close_b, beta, window,
                  entry=2.0, exit=0.5):
    """Z-score mean-reversion on a cointegrated pair spread.

    Builds a synthetic equity series for the spread portfolio (long A,
    short beta*B) so the existing run_engine can backtest the pair as a
    single instrument WITHOUT modification. The engine's look-ahead
    protection (single shift(1)) and trade-cost accounting stay intact.

    Why a synthetic EQUITY series, not the raw spread:
        run_engine computes returns as Close / Close.shift(1) - 1. The
        raw spread A - beta*B can be zero or negative, where pct_change
        is undefined / sign-flipping garbage. Instead we feed the engine
        a price series built from spread RETURNS:
            leg_return  = ret_A - beta * ret_B
            synth_price = (1 + leg_return).cumprod()   # always > 0
        synth_price.pct_change() recovers leg_return exactly, so the
        engine sees precisely the spread-portfolio return with nothing
        lost. Position direction is returned separately; the engine
        applies shift(1) and charges costs on sign changes.

    Args:
        close_a: Daily closing prices of asset A (long leg) as a Series.
        close_b: Daily closing prices of asset B (hedge leg) as a Series.
        beta: Hedge ratio from OLS (A = alpha + beta*B). STATIC here —
            computed on the full window, so it look-aheads. Acceptable
            for proof-of-concept (an optimistically-biased result that
            still failing means rolling-beta won't save it). Replace
            with rolling/Kalman beta before delivery.
        window: Rolling window for the z-score mean/std, in days.
            Use the pair's OU half-life as the natural choice.
        entry: Absolute z-score to open a position. Defaults to 2.0.
        exit: Absolute z-score to close back to flat. Defaults to 0.5.

    Returns:
        synth_price: Synthetic equity series of the spread portfolio,
            same index as the aligned pair. Feed as `close` to run_engine.
        position: Series of +1 (long spread) / 0 (flat) / -1 (short
            spread), same index. Feed as `position` to run_engine.

    Notes:
        z > +entry  -> spread rich  -> SHORT spread (-1), expect z -> 0.
        z < -entry  -> spread cheap -> LONG  spread (+1).
        |z| < exit  -> close to flat (0).
        The z-score window uses only trailing data (rolling), so the
        SIGNAL has no look-ahead. The only look-ahead is in beta, by
        deliberate POC choice — kept isolated so the two effects don't
        mix and confuse the result.
        Sequential state machine (not vectorised): today's position
        depends on whether we were already in a trade, like the stop /
        breakout strategies.
    """
    # Align on common dates (each instrument has its own gaps).
    df = pd.concat([close_a, close_b], axis=1,
                   keys=["A", "B"], sort=True).dropna()
    a = df["A"]
    b = df["B"]

    # --- Synthetic spread-portfolio equity (engine-safe "price") ---
    # beta is the hedge ratio on PRICE LEVELS (dollars of A per dollar
    # of B). It must NOT be applied raw to returns: ret_A - beta*ret_B
    # with a large beta (e.g. Crude/Gasoline beta ~50, different price
    # scales) drives leg_return below -1, so (1+leg).cumprod() flips
    # negative and the equity series explodes. That is a units error:
    # dollar-neutrality != return-neutrality.
    #
    # Correct dollar-neutral spread portfolio: long $w_a of A, short
    # $w_b of B, with weights proportional to the hedge ratio and
    # normalised by total capital deployed so the portfolio return is
    # well-scaled:
    #     w_a = 1 / (1 + |beta|),  w_b = |beta| / (1 + |beta|)
    #     leg_return = w_a * ret_a - w_b * ret_b
    ret_a = a.pct_change()
    ret_b = b.pct_change()
    w_a = 1.0 / (1.0 + abs(beta))
    w_b = abs(beta) / (1.0 + abs(beta))
    leg_return = w_a * ret_a - w_b * ret_b     # daily spread return
    synth_price = (1 + leg_return).cumprod()   # > 0 by construction
    synth_price.iloc[0] = 1.0                  # NaN from first pct_change

    # --- Z-score of the price spread (signal lives here) ---
    spread = a - beta * b
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z = (spread - mean) / std

    # --- Sequential state machine: -1 / 0 / +1 ---
    position = pd.Series(0.0, index=df.index)
    state = 0

    for i in range(len(df)):
        zi = z.iloc[i]
        if np.isnan(zi):          # window not warmed up yet
            position.iloc[i] = 0.0
            continue

        if state == 0:
            if zi > entry:
                state = -1        # spread rich -> short spread
            elif zi < -entry:
                state = 1         # spread cheap -> long spread
        else:                     # in a trade: exit when |z| shrinks
            if abs(zi) < exit:
                state = 0

        position.iloc[i] = float(state)

    return synth_price, position


def run_pair(name_a, ticker_a, name_b, ticker_b, beta, window,
             trade_start, end, load_start=None,
             entry=2.0, exit=0.5, cost=0.001):
    """Load a pair, build the spread strategy, run it through run_engine.

    Thin wrapper that bridges the two-series pair to the single-series
    run_engine contract. Does NOT touch the engine.

    Args:
        name_a, ticker_a: Label and yfinance ticker of asset A (long leg).
        name_b, ticker_b: Label and yfinance ticker of asset B (hedge).
        beta: Static hedge ratio (see zscore_spread).
        window: Z-score rolling window in days (pair half-life).
        trade_start: First date counted in the equity/metrics.
        end: End date (exclusive, yfinance convention).
        load_start: Data load start. Defaults to 3 years before
            trade_start so the z-window warms up before trading.
        entry, exit: Z-score thresholds. Defaults 2.0 / 0.5.
        cost: Per-trade cost passed to run_engine. Defaults to 0.001.

    Returns:
        result: Per-day DataFrame from run_engine.
        total_return: Total return over the traded window.
        max_dd: Worst drawdown over the traded window.
    """
    if load_start is None:
        load_start = (
            pd.Timestamp(trade_start) - pd.DateOffset(years=3)
        ).strftime("%Y-%m-%d")

    close_a = load_data(ticker_a, load_start, end)
    close_b = load_data(ticker_b, load_start, end)

    synth_price, position = zscore_spread(
        close_a, close_b, beta=beta, window=window,
        entry=entry, exit=exit)

    return run_engine(synth_price, position, trade_start, cost)