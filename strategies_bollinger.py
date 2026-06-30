import pandas as pd


def donchian_bollinger_b(close, pairs=None, threshold=0.5,
                         entry=20, exit_period=10,
                         bb_period=20, bb_std=2.0, pctb_entry=1.0,
                         pctb_exit=0.5,
                         target_vol=0.15, vol_window=30, max_pos=2.0):
    """Donchian breakout confirmed by Bollinger %b momentum + vol targeting.

    Momentum interpretation of %b (NOT mean-reversion): a high %b means
    price is pushing through the upper band — confirmation of strength,
    not an overbought signal. Aligns with the Turtle breakout logic
    instead of fighting it.

    Entry conditions (all must hold):
        1. Price breaks N-day high (Donchian breakout signal).
        2. %b >= pctb_entry (price at/above upper Bollinger band).
        3. Majority of EMA pairs are bullish (macro trend filter).

    Exit (either):
        - Price breaks M-day low (Donchian exit channel), OR
        - %b falls below pctb_exit (momentum faded back toward mean).

    Args:
        close: Daily closing prices as a pandas Series.
        pairs: EMA (fast, slow) pairs for ensemble vote. Defaults to
            [(5,20), (10,40), (20,80), (40,160), (64,256)].
        threshold: Minimum bullish vote fraction to allow entry.
            Defaults to 0.5.
        entry: Donchian entry look-back in days. Defaults to 20.
        exit_period: Donchian exit look-back in days. Defaults to 10.
        bb_period: SMA period for Bollinger Bands. Defaults to 20.
        bb_std: Number of standard deviations for bands. Defaults to 2.0.
        pctb_entry: %b level required to confirm entry. Defaults to 1.0
            (price at the upper band).
        pctb_exit: %b level below which momentum exit fires.
            Defaults to 0.5 (price back at the middle band).
        target_vol: Target annualised volatility. Defaults to 0.15.
        vol_window: Rolling window for volatility estimate. Defaults to 30.
        max_pos: Maximum position size / leverage cap. Defaults to 2.0.

    Returns:
        position: Fractional Series (vol-scaled 0/1), same index as close.

    Notes:
        %b = (Close - LowerBand) / (UpperBand - LowerBand).
        Not bounded to [0, 1]: in strong moves %b > 1 is normal — that
        is precisely the momentum signal we want, not an anomaly.
        Bands widen AFTER volatility rises, so %b lags at the very start
        of a move; the Donchian breakout is the leading trigger and %b
        is the confirming filter, not the other way round.
        Untested as of this writing — run against energy and agri
        instruments and compare maxDD / Calmar vs donchian_ensemble_pyramid.
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

    # 3. Bollinger %b (shifted: use yesterday's bands, no look-ahead)
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    upper = (sma + bb_std * std).shift(1)
    lower = (sma - bb_std * std).shift(1)
    pct_b = (close - lower) / (upper - lower)

    # 4. Sequential state machine
    raw_position = pd.Series(0.0, index=close.index)
    in_position = False

    for i in range(len(close)):
        price = close.iloc[i]
        b = pct_b.iloc[i]

        if in_position:
            momentum_faded = (not pd.isna(b)) and b < pctb_exit
            if price < low_m.iloc[i] or momentum_faded:
                in_position = False
            else:
                raw_position.iloc[i] = 1.0
        else:
            breakout = price > high_n.iloc[i]
            confirmed = (not pd.isna(b)) and b >= pctb_entry
            if breakout and confirmed and macro_bullish.iloc[i]:
                in_position = True
                raw_position.iloc[i] = 1.0

    # 5. Vol targeting
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return raw_position * size


def bollinger_squeeze(close, bb_period=20, bb_std=2.0,
                      kc_period=20, kc_mult=2.5,
                      entry=20, exit_period=10,
                      squeeze_window=10,
                      target_vol=0.15, vol_window=30, max_pos=2.0):
    """Volatility-squeeze breakout (TTM Squeeze style) + Donchian trigger.

    Detects abnormally low volatility — Bollinger Bands contracting
    INSIDE Keltner Channels — as a "loaded spring" state, then enters
    on the directional breakout that releases it.

    The squeeze itself is direction-agnostic: it only says a move is
    brewing. Direction comes from the Donchian N-day high breakout.
    This is the one Bollinger signal orthogonal to the EMA ensemble:
    it measures VOLATILITY STATE, not price direction.

    Entry conditions (all must hold):
        1. Squeeze was ON within the last `squeeze_window` days
           (volatility was compressed — spring loaded).
        2. Price breaks N-day high (Donchian breakout — direction).

    Exit: price breaks M-day low (Donchian exit channel).

    Math:
        Bollinger:  Middle = SMA(close, bb_period)
                    Upper/Lower = Middle ± bb_std * std(close, bb_period)
        Keltner:    Middle_K = EMA(close, kc_period)
                    ATR = mean(|close - close.shift(1)|, kc_period)
                    Upper_K/Lower_K = Middle_K ± kc_mult * ATR
        Squeeze ON: (Upper < Upper_K) AND (Lower > Lower_K)
            -> BB sits inside Keltner. Because std penalises large moves
               quadratically while ATR is linear, std collapses faster
               than ATR when the market goes quiet, so the BB band
               contracts inside the Keltner band. Self-normalising:
               no per-instrument threshold needed.

    Args:
        close: Daily closing prices as a pandas Series.
        bb_period: SMA/std period for Bollinger Bands. Defaults to 20.
        bb_std: Std-dev multiple for Bollinger Bands. Defaults to 2.0.
        kc_period: EMA/ATR period for Keltner Channels. Defaults to 20.
        kc_mult: ATR multiple for Keltner Channels. Defaults to 1.5.
        entry: Donchian entry look-back in days. Defaults to 20.
        exit_period: Donchian exit look-back in days. Defaults to 10.
        squeeze_window: Days to look back for a recent squeeze when
            confirming entry. Defaults to 10.
        target_vol: Target annualised volatility. Defaults to 0.15.
        vol_window: Rolling window for volatility estimate. Defaults to 30.
        max_pos: Maximum position size / leverage cap. Defaults to 2.0.

    Returns:
        position: Fractional Series (vol-scaled 0/1), same index as close.

    Notes:
        std and ATR both shifted by 1 day before use is NOT needed here
        because the Donchian channel already uses shift(1); the squeeze
        flag is computed on same-day bands but only consumed as a
        look-back condition (was-squeezed-recently), so no look-ahead:
        entry still requires today's price to exceed yesterday's N-day
        high.
        Untested as of writing — run against energy (NG, CL) and a
        trending metal (Gold) first; compare maxDD / Calmar vs
        donchian_ensemble_pyramid.
    """
    # --- Слой 1: Bollinger Bands (волатильность через std) ---
# --- Слой 1: Bollinger на дневной волатильности (не на уровне цены) ---
    bb_mid = close.rolling(bb_period).mean()
    daily_move = close - close.shift(1)
    bb_sd = daily_move.rolling(bb_period).std()      # σ дневных ходов
    bb_upper = bb_mid + bb_std * bb_sd
    bb_lower = bb_mid - bb_std * bb_sd

    # --- Слой 2: Keltner Channels (волатильность через ATR) ---
    kc_mid = close.ewm(span=kc_period, adjust=False).mean()
    atr = (close - close.shift(1)).abs().rolling(kc_period).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    # --- Слой 3: Squeeze ON = BB внутри Keltner ---
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    # «Был ли squeeze взведён за последние squeeze_window дней»
    recent_squeeze = squeeze_on.rolling(squeeze_window).max().astype(bool)

    # --- Слой 4: Donchian channel (directional trigger, shift против look-ahead) ---
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()

    # --- Слой 5: state machine ---
    raw_position = pd.Series(0.0, index=close.index)
    in_position = False

    """print(f"bb_sd:    {bb_sd.mean():.4f}")
    print(f"atr:      {atr.mean():.4f}")
    print(f"bb полуширина (bb_std*bb_sd):  {(bb_std*bb_sd).mean():.4f}")
    print(f"kc полуширина (kc_mult*atr):   {(kc_mult*atr).mean():.4f}")
    print(f"bb_mid:   {bb_mid.mean():.4f}")
    print(f"kc_mid:   {kc_mid.mean():.4f}")
    print(f"squeeze_on True: {squeeze_on.sum()} из {len(squeeze_on)}")"""

    for i in range(len(close)):
        price = close.iloc[i]
        if in_position:
            if price < low_m.iloc[i]:
                in_position = False
            else:
                raw_position.iloc[i] = 1.0
        else:
            loaded = bool(recent_squeeze.iloc[i])
            breakout = price > high_n.iloc[i]
            if loaded and breakout:
                in_position = True
                raw_position.iloc[i] = 1.0

    # --- Слой 6: vol targeting ---
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (252 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)
    return raw_position * size