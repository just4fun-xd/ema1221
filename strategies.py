import pandas as pd
import numpy as np


def ema_cross(close, fast=12, slow=21):
    """EMA crossover: long when fast EMA is above slow EMA.

    Baseline strategy. Simple but fragile — whipsaws in sideways
    markets, result varies wildly across years and instruments.

    Args:
        close: Daily closing prices as a pandas Series.
        fast: Period of the fast EMA. Defaults to 12.
        slow: Period of the slow EMA. Defaults to 21.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.

    Notes:
        Tested result: no consistent edge on commodities except gold in
        trending years. Drawdown exceeds 40% on Natural Gas and Cocoa.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return (ema_fast > ema_slow).astype(int)


def ema_cross_stop(close, fast=12, slow=21, stop=0.10):
    """EMA crossover with a fixed percentage stop-loss.

    Sequential logic — tracks entry price day by day and exits if price
    falls more than `stop` fraction below the entry level.

    Args:
        close: Daily closing prices as a pandas Series.
        fast: Period of the fast EMA. Defaults to 12.
        slow: Period of the slow EMA. Defaults to 21.
        stop: Exit threshold below entry price (0.10 = -10%).
            Defaults to 0.10.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.

    Notes:
        Tested result: stop fires rarely because the crossover signal
        exits first. Does not help the primary failure mode (many small
        losses in choppy markets, not one deep loss).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    bull = ema_fast > ema_slow

    position = pd.Series(0, index=close.index)
    entry_price = 0.0
    in_position = False

    for i in range(len(close)):
        price = close.iloc[i]
        if in_position:
            if price <= entry_price * (1 - stop) or not bull.iloc[i]:
                in_position = False
            else:
                position.iloc[i] = 1
        else:
            if bull.iloc[i]:
                in_position = True
                entry_price = price
                position.iloc[i] = 1
    return position


def sma_trend(close, period=200):
    """Trend-filter benchmark: long when price is above its SMA.

    Single slow signal, very few trades per year. Useful as a benchmark
    — simpler than any crossover but competitive on trending instruments.

    Args:
        close: Daily closing prices as a pandas Series.
        period: SMA look-back window in trading days. Defaults to 200.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.

    Notes:
        On gold, SMA(200) matched or beat EMA(12/21) in trending years
        (2024: +27.5% vs +12.4%). Lags badly in choppy years.
    """
    sma = close.rolling(period).mean()
    return (close > sma).astype(int)


def ema_trend(close, period=200):
    """Trend-filter using EMA instead of SMA.

    Reacts slightly faster than SMA of the same period. Kept as an
    alternative benchmark alongside sma_trend.

    Args:
        close: Daily closing prices as a pandas Series.
        period: EMA span in trading days. Defaults to 200.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.
    """
    ema = close.ewm(span=period, adjust=False).mean()
    return (close > ema).astype(int)


def ema_ensemble(close, pairs=None, threshold=0.5):
    """Ensemble of EMA crossovers across multiple timeframes.

    Each pair votes 1 (bullish) or 0 (bearish). Enter when the share
    of bullish votes exceeds `threshold`. Pairs follow a 1:4 geometric
    ladder (Carver, Systematic Trading).

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: List of (fast, slow) EMA period tuples. Defaults to
            [(5,20), (10,40), (20,80), (40,160), (64,256)].
        threshold: Minimum fraction of bullish pairs to enter (0–1).
            Defaults to 0.5.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.

    Notes:
        Stable to parameter choice (threshold 0.5–0.6, pair set).
        Cuts drawdowns vs single pair but also cuts peak returns.
        On its own, does not create edge — only manages risk.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)

    return (votes.mean(axis=1) > threshold).astype(int)


def ema_ensemble_voltarget(close, pairs=None, threshold=0.5,
                           target_vol=0.15, vol_window=30, max_pos=2.0):
    """Ensemble EMA signal with volatility-based position sizing.

    Direction from ema_ensemble; position size scaled so realised
    annual volatility matches `target_vol`. Caps leverage at `max_pos`.

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA period pairs. See ema_ensemble. Defaults to None.
        threshold: Bullish vote fraction to enter. Defaults to 0.5.
        target_vol: Target annualised volatility (0.15 = 15%).
            Defaults to 0.15.
        vol_window: Rolling window for volatility estimate in days.
            Defaults to 30.
        max_pos: Maximum position size (leverage cap). Defaults to 2.0.

    Returns:
        position: Fractional Series (not just 0/1), same index as close.

    Notes:
        Passes DD < 40% on all 8 instruments across 2021–2025.
        Max observed drawdown: -21.2%. Vol targeting scales both risk
        and return proportionally — does not create profit on its own.
    """
    direction = ema_ensemble(close, pairs=pairs, threshold=threshold)
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return direction * size


def ema_ensemble_long_short(close, pairs=None, threshold=0.2):
    """Ensemble EMA with long and short positions.

    Votes are +1 (fast > slow) or -1 (fast < slow). Enter long when
    average vote > threshold, short when < -threshold.

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA period pairs. Defaults to None (uses standard ladder).
        threshold: Score boundary for entry (symmetric). Defaults to 0.2.

    Returns:
        position: Series of +1 (long) / 0 (flat) / -1 (short).

    Notes:
        INEFFECTIVE on commodities. Structural upward bias of commodity
        markets makes systematic shorting unprofitable. Documented as a
        tested-and-rejected approach.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = np.where(ema_fast > ema_slow, 1, -1)

    score = votes.mean(axis=1)
    direction = pd.Series(0, index=close.index)
    direction[score > threshold] = 1
    direction[score < -threshold] = -1
    return direction


def ema_ensemble_voltarget_ls(close, pairs=None, threshold=0.5,
                              target_vol=0.15, vol_window=30, max_pos=2.0):
    """Ensemble EMA long/short with volatility-based position sizing.

    Combines ema_ensemble_long_short with vol targeting. Formally passes
    DD < 40% (max −26.7%), but only because vol targeting shrinks
    positions — not because shorts add value.

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA period pairs. Defaults to None.
        threshold: Score boundary for entry. Defaults to 0.5.
        target_vol: Target annualised volatility. Defaults to 0.15.
        vol_window: Rolling volatility window in days. Defaults to 30.
        max_pos: Leverage cap. Defaults to 2.0.

    Returns:
        position: Fractional Series (+/- allowed), same index as close.

    Notes:
        Shorts pass DD threshold only via position scaling, not signal
        quality. Profitable years: similar to long-only version.
    """
    direction = ema_ensemble_long_short(
        close, pairs=pairs, threshold=threshold)
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return direction * size


def ema_ensemble_voltarget_smart_ls(close, pairs_long=None, pairs_short=None, 
                                    threshold_long=0.2, threshold_short=-0.8,
                                    target_vol=0.15, vol_window=30, max_pos=2.0):
    """
    Асимметричный Long/Short ансамбль с умным таргетированием волатильности.
    """
    if pairs_long is None:
        pairs_long = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    if pairs_short is None:
        pairs_short = [(5, 20), (10, 40), (20, 80)]

    votes_long = []
    for fast, slow in pairs_long:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes_long.append(np.where(ema_fast > ema_slow, 1, -1))
    
    score_long = np.mean(votes_long, axis=0)
    
    votes_short = []
    for fast, slow in pairs_short:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes_short.append(np.where(ema_fast > ema_slow, 1, -1))
        
    score_short = np.mean(votes_short, axis=0)

    direction = pd.Series(0.0, index=close.index)
    
    direction[score_long > threshold_long] = 1.0
    
    direction[score_short < threshold_short] = -1.0
    
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * np.sqrt(252)
    
    raw_size = target_vol / annual_vol

    
    final_position = pd.Series(0.0, index=close.index)
    
    for i in range(len(close)):
        if pd.isna(annual_vol.iloc[i]):
            continue
            
        current_dir = direction.iloc[i]
        calculated_size = raw_size.iloc[i]
        
        if current_dir == 1.0:
            size = min(calculated_size, max_pos)
            final_position.iloc[i] = size
        elif current_dir == -1.0:

            size = max(min(calculated_size, 1.0), 0.5)
            final_position.iloc[i] = -size

    return final_position
