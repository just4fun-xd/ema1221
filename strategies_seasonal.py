import pandas as pd


def seasonal_gas(close, buy_months=(8, 9, 10, 11),
                 sell_months=(3, 4, 5, 6)):
    """Seasonal calendar strategy for Natural Gas.

    Holds long position only during historically strong months
    (pre-winter injection season). No price signal — pure calendar.

    Args:
        close: Daily closing prices as a pandas Series with DatetimeIndex.
        buy_months: Tuple of months (1-12) to be long. Defaults to
            (8, 9, 10, 11) — August through November (injection season).
        sell_months: Tuple of months to be flat. Defaults to
            (3, 4, 5, 6) — spring draw-down season.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.

    Notes:
        Hypothesis: gas rises before winter (storage injection demand),
        falls in spring (low demand after heating season).
        Test against Donchian Pyramid on the same instrument.
        If profitable in DIFFERENT years — valuable for combination.
        Key result (2019-2025): +36.8% in 2020, +62.9% in 2024,
        +59.3% in 2025. Exception: 2022 (-18.8%) due to geopolitical
        disruption (European gas crisis) overriding seasonal pattern.
    """
    position = pd.Series(0, index=close.index)
    position[close.index.month.isin(buy_months)] = 1
    return position


def donchian_seasonal(close, entry=20, exit_period=10,
                      buy_months=(8, 9, 10, 11)):
    """Donchian breakout filtered by seasonal calendar window.

    Combines two independent signals: (1) calendar filter — only enters
    during historically strong months; (2) price breakout — only enters
    when price breaks the N-day high. Both must hold simultaneously.
    Exits on Donchian low breakout regardless of season.

    This is structurally different from donchian_seasonal_voltarget:
    position size is binary (1 or 0), not volatility-scaled.

    Args:
        close: Daily closing prices as a pandas Series with DatetimeIndex.
        entry: Look-back window for entry breakout in days. Defaults to 20.
        exit_period: Look-back window for exit breakout in days.
            Must be less than entry. Defaults to 10.
        buy_months: Tuple of months (1-12) when entries are allowed.
            Defaults to (8, 9, 10, 11) — injection season for Natural Gas.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.

    Notes:
        Entry logic: price > 20-day high AND current month in buy_months.
        Exit logic: price < 10-day low (no seasonal constraint on exit).
        shift(1) applied before rolling to avoid look-ahead bias —
        channel boundaries use only yesterday's data.
        Key result vs raw Donchian (Natural Gas 2019-2025):
            - 2022: -12.7% vs -24.9% raw (seasonal filter avoided
              worst entries during European gas crisis peak)
            - 2023: -19.4% vs -48.1% raw (same protection)
            - 2024: +44.8% vs +56.8% raw (missed some late-year gains)
        Trade-off: captures seasonal trend strength, misses breakouts
        that occur outside the seasonal window (e.g. winter rallies).
    """
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()

    position = pd.Series(0, index=close.index)
    in_position = False

    for i in range(len(close)):
        price = close.iloc[i]
        if in_position:
            if price < low_m.iloc[i]:
                in_position = False
            else:
                position.iloc[i] = 1
        else:
            month_ok = close.index[i].month in buy_months
            if price > high_n.iloc[i] and month_ok:
                in_position = True
                position.iloc[i] = 1
    return position


def donchian_seasonal_voltarget(close, entry=20, exit_period=10,
                                buy_months=(8, 9, 10, 11),
                                target_vol=0.15, vol_window=30,
                                max_pos=2.0):
    """Donchian seasonal with volatility-based position sizing.

    Same entry/exit signal as donchian_seasonal (calendar-filtered
    breakout), but position size scaled by target_vol / realised_vol.
    Reduces exposure during high-volatility periods (gas price spikes).

    Implements the standard vol targeting formula:
        size = target_vol / annualised_vol, capped at max_pos.

    Args:
        close: Daily closing prices as a pandas Series.
        entry: Donchian entry look-back in days. Defaults to 20.
        exit_period: Donchian exit look-back in days. Defaults to 10.
        buy_months: Months when entries are allowed. Defaults to
            (8, 9, 10, 11).
        target_vol: Target annualised volatility (0.15 = 15%).
            Defaults to 0.15.
        vol_window: Rolling window for volatility estimate in days.
            Defaults to 30.
        max_pos: Maximum position size / leverage cap. Defaults to 2.0.

    Returns:
        position: Fractional Series (vol-scaled 0–max_pos),
            same index as close.

    Notes:
        Reuses donchian_seasonal for signal — vol targeting is a
        separate layer, not entangled with entry/exit logic.
        Key result vs donchian_seasonal (Natural Gas 2019-2025):
            Max drawdown reduced from -25.9% to -7.9%.
            Returns proportionally reduced: +44.8% → +9.5% in 2024.
            Return/drawdown ratio similar or better in most years.
        Vol targeting scales both risk AND return by the same factor.
        Best choice when combining with other strategies in a portfolio
        (equal risk contribution per instrument).
    """
    raw = donchian_seasonal(close, entry=entry,
                            exit_period=exit_period,
                            buy_months=buy_months)

    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return raw * size
