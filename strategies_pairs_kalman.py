import numpy as np
import pandas as pd

from engine import load_data, run_engine


def kalman_beta(a, b, delta=1e-4, R=1e-3):
    """Online hedge ratio beta[t] (and alpha[t]) via a Kalman filter.

    Estimates a time-varying linear relationship A[t] = beta[t]*B[t] +
    alpha[t], where the state [beta, alpha] follows a random walk. The
    estimate at day t uses ONLY data up to and including day t — no
    look-ahead — which is the whole point versus a static OLS beta
    computed on the full window.

    Why this matters: a cointegration relationship is not fixed. When it
    breaks (structural shift — e.g. the 2022 energy crisis dislocating
    crack spreads), a static OLS beta sits against a spread that has run
    away and will not revert, producing systematic losses. The Kalman
    beta adapts to the new relationship instead.

    Two free parameters trade off stability vs adaptiveness (the source
    of the well-known sensitivity of Kalman pairs trading):
        delta: process-noise scale. Larger -> beta moves faster, noisier.
        R: observation-noise variance. Larger -> trust new data less,
           beta smoother.
    Defaults (delta=1e-4, R=1e-3) are the common QuantStart values.

    Args:
        a: Price Series of asset A (long leg), aligned to b.
        b: Price Series of asset B (hedge leg), aligned to a.
        delta: Process-noise scale. Defaults to 1e-4.
        R: Observation-noise variance. Defaults to 1e-3.

    Returns:
        beta: Series of daily hedge ratios beta[t], same index as a.
    """
    n = len(a)
    beta_out = np.zeros(n)
    P = np.zeros((2, 2))                 # state covariance
    x = np.zeros(2)                      # state [beta, alpha]
    Vw = delta / (1 - delta) * np.eye(2)  # process-noise covariance

    for t in range(n):
        if t > 0:
            P = P + Vw                   # predict: inflate uncertainty

        F = np.array([b.iloc[t], 1.0])   # observation row [B, 1]
        yhat = F @ x                     # predict A from PAST state
        e = a.iloc[t] - yhat             # innovation (forecast error)
        S = F @ P @ F + R                # innovation variance
        K = (P @ F) / S                  # Kalman gain
        x = x + K * e                    # update with today's price
        P = P - np.outer(K, F) @ P
        beta_out[t] = x[0]

    return pd.Series(beta_out, index=a.index)


def zscore_spread_kalman(close_a, close_b, window,
                         entry=2.0, exit=0.5,
                         delta=1e-4, R=1e-3):
    """Z-score mean-reversion on a pair spread with Kalman hedge ratio.

    Same structure as zscore_spread, but beta is estimated online by a
    Kalman filter (no look-ahead) instead of a single static OLS fit on
    the full window. Directly addresses risk #1 of the project's own
    Karpathy-council list ("beta look-ahead ... rolling-beta before
    delivery").

    The synthetic equity series stays engine-safe exactly as before:
    a dollar-neutral spread portfolio whose daily return is well-scaled,
    so (1 + leg_return).cumprod() never goes non-positive. The engine
    keeps its single shift(1) look-ahead guard and charges costs on
    sign changes. Direction (+1/0/-1) is returned for the engine.

    Args:
        close_a: Daily closing prices of asset A (long leg) as a Series.
        close_b: Daily closing prices of asset B (hedge leg) as a Series.
        window: Rolling window for the z-score mean/std, in days.
        entry: Absolute z-score to open a position. Defaults to 2.0.
        exit: Absolute z-score to close back to flat. Defaults to 0.5.
        delta: Kalman process-noise scale. Defaults to 1e-4.
        R: Kalman observation-noise variance. Defaults to 1e-3.

    Returns:
        synth_price: Synthetic equity series of the spread portfolio.
            Feed as `close` to run_engine.
        position: Series of +1 (long spread) / 0 (flat) / -1 (short),
            same index. Feed as `position` to run_engine.

    Notes:
        beta[t] is time-varying, so the dollar-neutral weights
        w_a = 1/(1+|beta_t|), w_b = |beta_t|/(1+|beta_t|) are also
        recomputed each day. The z-score uses spread_t = A - beta_t*B,
        a rolling mean/std over `window` — trailing only, no look-ahead.
        Kalman beta is sensitive to delta/R; they are explicit args so
        the choice is never silent.
    """
    # Align on common dates.
    df = pd.concat([close_a, close_b], axis=1,
                   keys=["A", "B"], sort=True).dropna()
    a = df["A"]
    b = df["B"]

    # --- Time-varying hedge ratio (no look-ahead) ---
    beta = kalman_beta(a, b, delta=delta, R=R)

    # --- Dollar-neutral spread-portfolio equity (engine-safe) ---
    # Weights recomputed daily from beta[t]; same normalisation as the
    # static version so leg_return stays > -1 and cumprod stays positive.
    ret_a = a.pct_change()
    ret_b = b.pct_change()
    abs_beta = beta.abs()
    w_a = 1.0 / (1.0 + abs_beta)
    w_b = abs_beta / (1.0 + abs_beta)
    leg_return = w_a * ret_a - w_b * ret_b
    synth_price = (1 + leg_return).cumprod()
    synth_price.iloc[0] = 1.0

    # --- Z-score of the time-varying spread (signal) ---
    spread = a - beta * b
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z = (spread - mean) / std

    # --- Sequential state machine: -1 / 0 / +1 ---
    position = pd.Series(0.0, index=df.index)
    state = 0
    for i in range(len(df)):
        zi = z.iloc[i]
        if np.isnan(zi):
            position.iloc[i] = 0.0
            continue
        if state == 0:
            if zi > entry:
                state = -1
            elif zi < -entry:
                state = 1
        else:
            if abs(zi) < exit:
                state = 0
        position.iloc[i] = float(state)

    return synth_price, position
