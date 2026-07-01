import pandas as pd
import numpy as np


def dual_momentum_commodity(prices, native=None, lookback=126, top_frac=0.33,
                            rebalance="ME", skip=21, vol_window=None):
    """Cross-sectional dual momentum for a commodity futures basket.

    Deliberately DIFFERENT from the equity `dual_momentum`:

      - NO SMA200 absolute-trend filter. Commodities mean-revert; an
        SMA200 gate reproduces the documented lag failure (late entry /
        exit) seen across EMA200 regime filters. The short leg here is a
        genuine cross-sectional bet, not gated by absolute trend.
      - SYMMETRIC long/short, gross-neutral. Commodities have no
        structural up-drift, so the short leg can carry real spread alpha
        (a weak-carry / contango name shorted against a strong one),
        unlike equities where shorting a rising asset only caught bounces.
      - top_frac defaults to 0.33 (tertiles), not 0.2. With ~17 names a
        quintile is 3 per leg; tertiles give ~5-6 per leg, a less noisy
        cross-section.
      - native mask baked into weights. On a date where an instrument did
        not trade (market closed, forward-filled price), its weight is 0
        and it is excluded from the ranking cross-section. This keeps the
        engine ignorant of the mask -- it receives clean weights.

    Args:
        prices: DataFrame of daily close (columns = instruments), already
            union-calendar aligned and forward-filled (panel_close).
        native: Boolean DataFrame, same grid as prices. True where the
            instrument traded natively that day. None = treat all as
            native (equity-style behaviour, for A/B checks).
        lookback: Momentum window in trading days. 126 ~= 6 months.
        top_frac: Fraction of the eligible cross-section per leg.
            0.33 = tertiles.
        rebalance: Rebalance frequency ("ME" = month-end).
        skip: Days dropped at the end of the momentum window to avoid the
            short-term reversal effect. 21 ~= 1 month. 0 disables.
        vol_window: If set, weight within each leg is proportional to
            1/vol over this window (risk parity within leg). None = equal
            dollars per name.

    Returns:
        weights: DataFrame of target weights (same grid as prices). Longs
            sum ~= +1, shorts sum ~= -1, net ~= 0 on each rebalance.

    Notes:
        Vol-targeting / leverage is applied by run_portfolio downstream,
        not here -- this returns raw market-neutral tertile weights.
        The single shift(1) look-ahead guard lives in the engine; this
        function never shifts returns into signals.
    """
    past = prices.shift(lookback)
    recent = prices.shift(skip) if skip else prices
    mom = recent / past - 1

    if native is None:
        native = pd.DataFrame(True, index=prices.index,
                              columns=prices.columns)
    else:
        native = native.reindex_like(prices).fillna(False).astype(bool)

    if vol_window is not None:
        inv_vol = 1.0 / (prices.pct_change().rolling(vol_window).std()
                         * (252 ** 0.5))
        inv_vol = inv_vol.replace([np.inf, -np.inf], 0.0)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebal_dates = prices.resample(rebalance).last().index

    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        eff = sub.index[-1]

        # Eligible cross-section: momentum defined AND native on eff date.
        row = sub.iloc[-1]
        elig = row.dropna().index
        elig = [c for c in elig if bool(native.loc[eff, c])]
        row = row[elig]
        if len(row) < 5:
            continue

        n = max(1, int(len(row) * top_frac))
        ranked = row.sort_values(ascending=False)
        longs = list(ranked.head(n).index)
        shorts = list(ranked.tail(n).index)

        w = pd.Series(0.0, index=prices.columns)

        if vol_window is not None:
            iv = inv_vol.loc[eff].reindex(prices.columns).fillna(0.0)
            if longs:
                lw = iv[longs]
                w[longs] = (lw / lw.sum()).values if lw.sum() > 0 else 0.0
            if shorts:
                sw = iv[shorts]
                w[shorts] = -(sw / sw.sum()).values if sw.sum() > 0 else 0.0
        else:
            if longs:
                w[longs] = 1.0 / len(longs)
            if shorts:
                w[shorts] = -1.0 / len(shorts)

        weights.loc[dt:] = w.values

    # Bake native mask: zero weight on any day an instrument wasn't native,
    # so the engine never holds a position in a closed market.
    weights = weights.where(native, other=0.0)
    return weights


def dual_momentum_commodity_long_only(prices, native=None, lookback=126,
                                      top_frac=0.33, rebalance="ME", skip=21):
    """Long-only control for the commodity basket.

    Same ranking, top tier only, no short leg. The long-minus-short
    spread (market-neutral minus this) isolates whether the commodity
    short leg carries alpha -- the open question that equities answered
    'no' but commodities may answer differently (no structural up-drift).

    Args:
        prices: DataFrame of daily close (panel_close).
        native: Boolean DataFrame native mask, or None for all-native.
        lookback: Momentum window in trading days. Defaults to 126.
        top_frac: Long-leg fraction. Defaults to 0.33.
        rebalance: Rebalance frequency. Defaults to "ME".
        skip: Days dropped at window end. Defaults to 21.

    Returns:
        weights: DataFrame; longs sum to +1, no shorts.
    """
    past = prices.shift(lookback)
    recent = prices.shift(skip) if skip else prices
    mom = recent / past - 1

    if native is None:
        native = pd.DataFrame(True, index=prices.index,
                              columns=prices.columns)
    else:
        native = native.reindex_like(prices).fillna(False).astype(bool)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebal_dates = prices.resample(rebalance).last().index

    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        eff = sub.index[-1]
        row = sub.iloc[-1]
        elig = [c for c in row.dropna().index if bool(native.loc[eff, c])]
        row = row[elig]
        if len(row) < 5:
            continue
        n = max(1, int(len(row) * top_frac))
        longs = list(row.sort_values(ascending=False).head(n).index)
        w = pd.Series(0.0, index=prices.columns)
        if longs:
            w[longs] = 1.0 / len(longs)
        weights.loc[dt:] = w.values

    weights = weights.where(native, other=0.0)
    return weights
