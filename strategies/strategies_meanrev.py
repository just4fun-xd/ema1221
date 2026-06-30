import pandas as pd


def _compure_rsi(close, period=14):
    """Compute RSI as a helper — not a strategy itself.

    Args:
        close: Daily closing prices as a pandas Series.
        period: Look-back window for RSI. Defaults to 14.

    Returns:
        rsi: Series of RSI values (0-100), same index as close.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_rsi(close, bb_period=20, bb_std=2.0,
                  rsi_period=14, rsi_buy=30, rsi_exit=50):
    """Mean-reversion: buy when price oversold by both BB and RSI.

    Entry: price below lower Bollinger Band AND RSI below rsi_buy.
    Exit: price returns above SMA(bb_period) OR RSI rises above rsi_exit.
    Opposite logic to trend-following — profits from return to mean.

    Args:
        close: Daily closing prices as a pandas Series.
        bb_period: SMA period for Bollinger Bands. Defaults to 20.
        bb_std: Number of standard deviations for bands. Defaults to 2.0.
        rsi_period: RSI look-back window in days. Defaults to 14.
        rsi_buy: RSI level below which we enter (oversold). Defaults to 30.
        rsi_exit: RSI level above which we exit (momentum recovered).
            Defaults to 50.

    Returns:
        position: Series of 1 (in market) / 0 (cash), same index as close.

    Notes:
        Both signals must fire simultaneously to enter — reduces false
        signals vs using BB or RSI alone.
        Exit on SMA cross OR RSI recovery — whichever comes first.
        Mean-reversion works in range-bound markets; loses in strong
        sustained trends (keeps buying falling knives).
    """
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    lower_band = sma - bb_std * std

    rsi = _compure_rsi(close, rsi_period)

    oversold = (close < lower_band) & (rsi < rsi_buy)
    recovered = (close > sma) | (rsi > rsi_exit)

    position = pd.Series(0, index=close.index)
    in_position = False

    for i in range(len(close)):
        if in_position:
            if recovered.iloc[i]:
                in_position = False
            else:
                position.iloc[i] = 1
        elif oversold.iloc[i]:
            in_position = True
            position.iloc[i] = 1

    return position
