import pandas as pd


def donchian_breakout(close, entry=20, exit_period=10):
    """Donchian channel breakout — Turtle System 1 (long only).

    Enter long when price breaks the N-day high; exit when it breaks
    the M-day low. Reacts to raw price extremes, not smoothed averages.
    Fundamentally different signal type from EMA crossover.

    Args:
        close: Daily closing prices as a pandas Series.
        entry: Look-back window for entry breakout in days. Defaults to 20.
        exit_period: Look-back window for exit breakout in days.
            Must be less than entry. Defaults to 10.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.

    Notes:
        Shift(1) applied before rolling to avoid look-ahead: channel
        boundaries are calculated from yesterday's data only.
        Without risk management, drawdown exceeds 40% on Natural Gas
        (-55% in 2022) and Cocoa (-46% in 2025).
        Grains (wheat, corn) show more consistent moderate profits.
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
            if price > high_n.iloc[i]:
                in_position = True
                position.iloc[i] = 1
    return position


def donchian_breakout_ls(close, entry=20, exit_period=10):
    """Donchian channel breakout with long and short positions.

    INEFFECTIVE — documented as tested-and-rejected approach.
    Short selling against the structural upward bias of commodity
    markets produces worse drawdowns than long-only in most years.

    Args:
        close: Daily closing prices as a pandas Series.
        entry: Look-back window for entry breakout. Defaults to 20.
        exit_period: Look-back window for exit. Defaults to 10.

    Returns:
        position: Series of +1 (long) / 0 (flat) / -1 (short).

    Notes:
        Worst result: Natural Gas 2023 drawdown -71.3%.
        Cocoa 2024 drawdown -64.7% (while Cocoa was rising +108%).
    """
    high_entry = close.shift(1).rolling(entry).max()
    low_entry = close.shift(1).rolling(entry).min()
    high_exit = close.shift(1).rolling(exit_period).max()
    low_exit = close.shift(1).rolling(exit_period).min()

    position = pd.Series(0, index=close.index)
    current_pos = 0

    for i in range(len(close)):
        price = close.iloc[i]
        if current_pos == 1:
            if price < low_exit.iloc[i]:
                current_pos = 0
            else:
                position.iloc[i] = 1
        elif current_pos == -1:
            if price > high_exit.iloc[i]:
                current_pos = 0
            else:
                position.iloc[i] = -1
        else:
            if price > high_entry.iloc[i]:
                current_pos = 1
                position.iloc[i] = 1
            elif price < low_entry.iloc[i]:
                current_pos = -1
                position.iloc[i] = -1
    return position


def donchian_ensemble_voltarget(close, pairs=None, threshold=0.5,
                                entry=20, exit_period=10,
                                target_vol=0.15, vol_window=30,
                                max_pos=2.0):
    """Donchian breakout filtered by EMA ensemble + vol targeting.

    Entry requires both: (1) price breaks N-day high, AND (2) majority
    of EMA pairs are bullish (macro filter). Position sized by vol target.

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA (fast, slow) pairs for ensemble vote. Defaults to
            [(5,20), (10,40), (20,80), (40,160), (64,256)].
        threshold: Minimum bullish vote fraction to allow entry.
            Defaults to 0.5.
        entry: Donchian entry look-back in days. Defaults to 20.
        exit_period: Donchian exit look-back in days. Defaults to 10.
        target_vol: Target annualised volatility. Defaults to 0.15.
        vol_window: Rolling volatility window in days. Defaults to 30.
        max_pos: Leverage cap. Defaults to 2.0.

    Returns:
        position: Fractional Series, same index as close.

    Notes:
        Passes DD < 40% on all instruments (max -15.6%).
        Vol targeting dramatically tames Natural Gas
        (raw Donchian -55% DD → this strategy -7%).
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)

    macro_bullish = votes.mean(axis=1) > threshold
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()

    raw_position = pd.Series(0, index=close.index)
    in_position = False

    for i in range(len(close)):
        price = close.iloc[i]
        if in_position:
            if price < low_m.iloc[i]:
                in_position = False
            else:
                raw_position.iloc[i] = 1
        else:
            if price > high_n.iloc[i] and macro_bullish.iloc[i]:
                in_position = True
                raw_position.iloc[i] = 1

    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return raw_position * size


def donchian_ensemble_macd_voltarget(close, pairs=None, threshold=0.5,
                                     entry=20, exit_period=10,
                                     target_vol=0.15, vol_window=30,
                                     max_pos=2.0):
    """Donchian + EMA ensemble + MACD confirmation + vol targeting.

    Adds MACD histogram as a third entry filter on top of ensemble.
    Tested result: MACD adds almost nothing over the ensemble because
    both are EMA-based and highly correlated. Kept for documentation.

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA pairs for ensemble. Defaults to standard ladder.
        threshold: Bullish vote fraction. Defaults to 0.5.
        entry: Donchian entry window. Defaults to 20.
        exit_period: Donchian exit window. Defaults to 10.
        target_vol: Target annualised volatility. Defaults to 0.15.
        vol_window: Volatility rolling window. Defaults to 30.
        max_pos: Leverage cap. Defaults to 2.0.

    Returns:
        position: Fractional Series, same index as close.

    Notes:
        Results nearly identical to donchian_ensemble_voltarget.
        MACD redundant with ensemble — both detect EMA crossovers.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)

    macro_bullish = votes.mean(axis=1) > threshold

    macd_line = (close.ewm(span=12, adjust=False).mean()
                 - close.ewm(span=26, adjust=False).mean())
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    macd_bullish = (histogram > 0) & (histogram > histogram.shift(1))

    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()

    raw_position = pd.Series(0, index=close.index)
    in_position = False

    for i in range(len(close)):
        price = close.iloc[i]
        if in_position:
            if price < low_m.iloc[i]:
                in_position = False
            else:
                raw_position.iloc[i] = 1
        else:
            if (price > high_n.iloc[i]
                    and macro_bullish.iloc[i]
                    and macd_bullish.iloc[i]):
                in_position = True
                raw_position.iloc[i] = 1

    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return raw_position * size


def donchian_ensemble_macd_voltarget_ls(close, pairs=None, threshold=0.5,
                                        entry=20, exit_period=10,
                                        target_vol=0.15, vol_window=30,
                                        max_pos=2.0):
    """Donchian + EMA ensemble + MACD + vol targeting — long/short.

    INEFFECTIVE — documented as tested-and-rejected approach.
    Long/short version of donchian_ensemble_macd_voltarget. Formally
    passes DD < 40% (max -34.4%) due to vol sizing, not signal quality.

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA pairs. Defaults to standard ladder.
        threshold: Vote boundary for entry. Defaults to 0.5.
        entry: Donchian entry window. Defaults to 20.
        exit_period: Donchian exit window. Defaults to 10.
        target_vol: Target annualised volatility. Defaults to 0.15.
        vol_window: Volatility rolling window. Defaults to 30.
        max_pos: Leverage cap. Defaults to 2.0.

    Returns:
        position: Fractional Series (+/- allowed), same index as close.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)

    trend_score = votes.mean(axis=1)
    macro_bullish = trend_score > threshold
    macro_bearish = trend_score < (1 - threshold)

    macd_line = (close.ewm(span=12, adjust=False).mean()
                 - close.ewm(span=26, adjust=False).mean())
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    macd_bullish = (hist > 0) & (hist > hist.shift(1))
    macd_bearish = (hist < 0) & (hist < hist.shift(1))

    high_entry = close.shift(1).rolling(entry).max()
    low_entry = close.shift(1).rolling(entry).min()
    high_exit = close.shift(1).rolling(exit_period).max()
    low_exit = close.shift(1).rolling(exit_period).min()

    raw_position = pd.Series(0.0, index=close.index)
    current_state = 0

    for i in range(len(close)):
        price = close.iloc[i]
        if current_state == 1:
            if price < low_exit.iloc[i]:
                current_state = 0
            else:
                raw_position.iloc[i] = 1.0
        elif current_state == -1:
            if price > high_exit.iloc[i]:
                current_state = 0
            else:
                raw_position.iloc[i] = -1.0
        else:
            if (price > high_entry.iloc[i]
                    and macro_bullish.iloc[i]
                    and macd_bullish.iloc[i]):
                current_state = 1
                raw_position.iloc[i] = 1.0
            elif (price < low_entry.iloc[i]
                  and macro_bearish.iloc[i]
                  and macd_bearish.iloc[i]):
                current_state = -1
                raw_position.iloc[i] = -1.0

    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return raw_position * size


def donchian_ensemble_macd_pyramid(close, pairs=None, threshold=0.5,
                                   entry=20, exit_period=10,
                                   target_vol=0.15, vol_window=30,
                                   max_pos=2.0, pyramid_atr=1.5,
                                   atr_period=14):
    """Donchian + EMA ensemble + MACD + pyramiding + vol targeting.

    Two-stage entry: 50% on breakout, +50% when price rises
    pyramid_atr * ATR from entry. MACD as additional entry filter.
    See donchian_ensemble_pyramid for version without MACD.

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA pairs for ensemble. Defaults to standard ladder.
        threshold: Bullish vote fraction. Defaults to 0.5.
        entry: Donchian entry window. Defaults to 20.
        exit_period: Donchian exit window. Defaults to 10.
        target_vol: Target annualised volatility. Defaults to 0.15.
        vol_window: Volatility rolling window. Defaults to 30.
        max_pos: Leverage cap. Defaults to 2.0.
        pyramid_atr: Price rise in ATR units to trigger full position.
            Defaults to 1.5.
        atr_period: ATR look-back window. Defaults to 14.

    Returns:
        position: Fractional Series (0.5 on entry, 1.0 after pyramid).

    Notes:
        Results nearly identical to donchian_ensemble_pyramid (no MACD).
        MACD redundant — superseded by donchian_ensemble_pyramid.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)

    macro_bullish = votes.mean(axis=1) > threshold

    macd_line = (close.ewm(span=12, adjust=False).mean()
                 - close.ewm(span=26, adjust=False).mean())
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    macd_bullish = (histogram > 0) & (histogram > histogram.shift(1))

    atr = (close - close.shift(1)).abs().rolling(atr_period).mean()
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()

    raw_position = pd.Series(0.0, index=close.index)
    in_position = False
    pyramid_done = False
    entry_price = 0.0

    for i in range(len(close)):
        price = close.iloc[i]
        current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.0

        if in_position:
            if price < low_m.iloc[i]:
                in_position = False
                pyramid_done = False
                entry_price = 0.0
            else:
                if not pyramid_done and price >= (
                        entry_price + pyramid_atr * current_atr):
                    pyramid_done = True
                raw_position.iloc[i] = 1.0 if pyramid_done else 0.5
        elif (price > high_n.iloc[i]
              and macro_bullish.iloc[i]
              and macd_bullish.iloc[i]):
            in_position = True
            pyramid_done = False
            entry_price = price
            raw_position.iloc[i] = 0.5

    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return raw_position * size


def donchian_ensemble_pyramid(close, pairs=None, threshold=0.5,
                              entry=20, exit_period=10,
                              target_vol=0.15, vol_window=30,
                              max_pos=2.0, pyramid_atr=1.5,
                              atr_period=14):
    """Donchian breakout + EMA ensemble filter + pyramiding + vol targeting.

    Best result of the Donchian research track. Three complementary
    risk layers on top of the original Turtle breakout signal.

    Entry conditions (all must hold):
        1. Price breaks N-day high (Donchian breakout signal).
        2. Majority of EMA pairs are bullish (macro trend filter).

    Position sizing:
        - 50% on entry (reduced exposure to false breakouts).
        - +50% when price rises pyramid_atr * ATR above entry
          (confirms trend, adds to winner).
        - Full position scaled by target_vol / realised_vol.

    Exit: price breaks M-day low (Donchian exit channel).

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA (fast, slow) pairs for ensemble vote. Defaults to
            [(5,20), (10,40), (20,80), (40,160), (64,256)].
        threshold: Minimum bullish vote fraction to allow entry.
            Defaults to 0.5.
        entry: Donchian entry look-back in days. Defaults to 20.
        exit_period: Donchian exit look-back in days. Defaults to 10.
        target_vol: Target annualised volatility (e.g. 0.15 = 15%).
            Defaults to 0.15.
        vol_window: Rolling window for volatility estimate. Defaults to 30.
        max_pos: Maximum position size / leverage cap. Defaults to 2.0.
        pyramid_atr: Price rise in ATR multiples to trigger full size.
            Defaults to 1.5.
        atr_period: ATR calculation look-back. Defaults to 14.

    Returns:
        position: Fractional Series (0 / 0.5 / 1.0 * vol_scale),
            same index as close.

    Notes:
        No MACD — redundant with ensemble (both EMA-based, test showed
        near-identical results with and without MACD).
        Key result (2021–2025, 8 commodities):
            - Max drawdown: -14.0% (all instruments pass DD < 40%).
            - Profitable years: 58% (highest across all strategies).
            - Tamed Natural Gas from -55% raw Donchian DD to -6%.
            - Cocoa 2024: +38% / -8% (raw Donchian: +108% / -37%).
        Trade-off: pyramiding caps peak returns vs raw Donchian in
        very strong trend years.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    # 1. Macro filter: EMA ensemble vote
    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)

    macro_bullish = votes.mean(axis=1) > threshold

    # 2. Donchian channel (shifted to avoid look-ahead)
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()

    # 3. ATR for pyramid trigger
    atr = (close - close.shift(1)).abs().rolling(atr_period).mean()

    # 4. Sequential state machine
    raw_position = pd.Series(0.0, index=close.index)
    in_position = False
    pyramid_done = False
    entry_price = 0.0

    for i in range(len(close)):
        price = close.iloc[i]
        current_atr = atr.iloc[i]

        if in_position:
            if price < low_m.iloc[i]:
                in_position = False
                pyramid_done = False
                entry_price = 0.0
            else:
                if not pyramid_done and not pd.isna(current_atr):
                    if price >= entry_price + pyramid_atr * current_atr:
                        pyramid_done = True
                raw_position.iloc[i] = 1.0 if pyramid_done else 0.5
        else:
            if price > high_n.iloc[i] and macro_bullish.iloc[i]:
                in_position = True
                pyramid_done = False
                entry_price = price
                raw_position.iloc[i] = 0.5

    # 5. Vol targeting
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return raw_position * size


