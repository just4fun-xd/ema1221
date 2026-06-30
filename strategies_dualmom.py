import pandas as pd
import numpy as np


def dual_momentum(prices, lookback=126, top_frac=0.2,
                  sma_filter=200, rebalance="ME", skip=21):
    """Cross-sectional dual momentum, market-neutral.

    Each rebalance: rank the basket by momentum over `lookback`, go long
    the top `top_frac` against short the bottom `top_frac` in equal
    weights. A short is forbidden if the instrument's price is above its
    own SMA(`sma_filter`) — the absolute-trend leg of dual momentum.

    This is the correct form of the short leg: a bet on the *spread*
    between strong and weak names, not on the direction of any single
    instrument. Market beta cancels between the two legs, which is why it
    avoids the failure mode of the mirror-image directional short
    (shorting a structurally rising asset and catching the bounce).

    Args:
        prices: DataFrame of daily close prices (columns = instruments).
        lookback: Momentum window in trading days. 126 ~= 6 months.
        top_frac: Fraction of basket in each leg (0.2 = quintiles).
        sma_filter: Window of the absolute trend filter. None disables it.
        rebalance: Rebalance frequency ("ME" = month-end, "W" = weekly).
        skip: Days skipped at the end of the momentum window to avoid the
            short-term reversal effect. 21 ~= 1 month. 0 disables.

    Returns:
        weights: DataFrame of target weights (same grid as prices).
            Sum of longs ~= +1, sum of shorts ~= -1, net ~= 0.

    Notes:
        Daily data is sufficient — the signal is slow (weeks to months).
        Continuous futures for commodities must be roll-adjusted, else
        roll gaps create false momentum. Equity/ETF yfinance data is
        clean — validate here first, then move to futures (Databento).
    """
    # Momentum measured from (t - lookback) to (t - skip): standard
    # 12-1 style construction that drops the most recent month.
    past = prices.shift(lookback)
    recent = prices.shift(skip) if skip else prices
    mom = recent / past - 1

    if sma_filter is not None:
        sma = prices.rolling(sma_filter).mean()
        above_sma = prices > sma
    else:
        above_sma = pd.DataFrame(False, index=prices.index,
                                 columns=prices.columns)

    weights = pd.DataFrame(0.0, index=prices.index,
                           columns=prices.columns)
    rebal_dates = prices.resample(rebalance).last().index

    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        row = sub.iloc[-1].dropna()
        if len(row) < 5:
            continue

        n = max(1, int(len(row) * top_frac))
        ranked = row.sort_values(ascending=False)
        longs = list(ranked.head(n).index)
        shorts = list(ranked.tail(n).index)

        # Absolute filter: do not short an instrument still above SMA200.
        eff = sub.index[-1]
        shorts = [s for s in shorts if not bool(above_sma.loc[eff, s])]

        w = pd.Series(0.0, index=prices.columns)
        if longs:
            w[longs] = 1.0 / len(longs)
        if shorts:
            w[shorts] = -1.0 / len(shorts)

        weights.loc[dt:] = w.values

    return weights


def dual_momentum_long_only(prices, lookback=126, top_frac=0.2,
                            rebalance="ME", skip=21):
    """Long-only control: same ranking, top tier only, no short leg.

    Used to isolate what the short leg actually contributes. The
    long-minus-short spread (this minus the full market-neutral version)
    is the honest test of the short leg's value, especially in 2022.

    Args:
        prices: DataFrame of daily close prices.
        lookback: Momentum window in trading days. Defaults to 126.
        top_frac: Fraction of basket in the long leg. Defaults to 0.2.
        rebalance: Rebalance frequency. Defaults to "ME".
        skip: Days skipped at the end of the window. Defaults to 21.

    Returns:
        weights: DataFrame of weights, longs sum to +1, no shorts.
    """
    past = prices.shift(lookback)
    recent = prices.shift(skip) if skip else prices
    mom = recent / past - 1

    weights = pd.DataFrame(0.0, index=prices.index,
                           columns=prices.columns)
    rebal_dates = prices.resample(rebalance).last().index

    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        row = sub.iloc[-1].dropna()
        if len(row) < 5:
            continue
        n = max(1, int(len(row) * top_frac))
        longs = list(row.sort_values(ascending=False).head(n).index)
        w = pd.Series(0.0, index=prices.columns)
        w[longs] = 1.0 / len(longs)
        weights.loc[dt:] = w.values

    return weights


def _rank_legs(prices, lookback, top_frac, skip):
    """Shared ranking: returns (mom, rebal_dates) for the variants below.

    Computes 12-1 style momentum (lookback window, last `skip` days
    dropped) and the month-end rebalance dates. Factored out so the
    improved variants don't duplicate the ranking block.
    """
    past = prices.shift(lookback)
    recent = prices.shift(skip) if skip else prices
    mom = recent / past - 1
    return mom


def dual_momentum_tilt(prices, benchmark=None, lookback=126, top_frac=0.2,
                       sma_filter=200, rebalance="ME", skip=21,
                       short_frac=0.5, market_sma=200):
    """Improvement 1: long-tilt with dynamic gross short exposure.

    Two changes vs the symmetric base:
      - Asymmetric legs: longs sum to +1.0, shorts to -short_frac
        (e.g. 0.5). Keeps most of the structural up-drift while still
        carrying a hedge.
      - Dynamic gross: when the broad market is in an uptrend
        (benchmark > its SMA(market_sma)), shrink the short leg further
        (halve it). Pay less for the hedge in bull years -- directly
        targets the negative bull-year spread.

    Args:
        prices: DataFrame of daily close prices.
        benchmark: Series of market proxy (e.g. SPY). None disables the
            dynamic dampening (then shorts stay at short_frac always).
        lookback: Momentum window in trading days. Defaults to 126.
        top_frac: Fraction of basket per leg. Defaults to 0.2.
        sma_filter: Per-instrument absolute trend filter. Defaults to 200.
        rebalance: Rebalance frequency. Defaults to "ME".
        skip: Days dropped at window end. Defaults to 21.
        short_frac: Base gross of the short leg (long is 1.0).
            Defaults to 0.5.
        market_sma: SMA window for the market-regime dampener.
            Defaults to 200.

    Returns:
        weights: DataFrame of weights; longs sum +1, shorts sum
            -short_frac (halved further in market uptrends).

    Notes:
        Net long by construction (+1 long, <=0.5 short) -> beta NOT near
        zero. This trades market-neutrality for less bull-year drag.
        Expect TEST 1 (beta~0) to FAIL by design -- that is the point.
    """
    mom = _rank_legs(prices, lookback, top_frac, skip)

    if sma_filter is not None:
        above_sma = prices > prices.rolling(sma_filter).mean()
    else:
        above_sma = pd.DataFrame(False, index=prices.index,
                                 columns=prices.columns)

    if benchmark is not None:
        bench = benchmark.reindex(prices.index).ffill()
        mkt_up = bench > bench.rolling(market_sma).mean()
    else:
        mkt_up = pd.Series(False, index=prices.index)

    weights = pd.DataFrame(0.0, index=prices.index,
                           columns=prices.columns)
    rebal_dates = prices.resample(rebalance).last().index

    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        row = sub.iloc[-1].dropna()
        if len(row) < 5:
            continue
        eff = sub.index[-1]

        n = max(1, int(len(row) * top_frac))
        ranked = row.sort_values(ascending=False)
        longs = list(ranked.head(n).index)
        shorts = [s for s in ranked.tail(n).index
                  if not bool(above_sma.loc[eff, s])]

        sf = short_frac
        if bool(mkt_up.reindex([eff]).iloc[0]):
            sf = short_frac * 0.5

        w = pd.Series(0.0, index=prices.columns)
        if longs:
            w[longs] = 1.0 / len(longs)
        if shorts:
            w[shorts] = -sf / len(shorts)
        weights.loc[dt:] = w.values

    return weights


def dual_momentum_regime(prices, benchmark, lookback=126, top_frac=0.2,
                         sma_filter=200, rebalance="ME", skip=21,
                         market_sma=200):
    """Improvement 2: short leg only switches on in risk-off regimes.

    Long leg is always on (+1). The short leg is added ONLY when the
    broad market is below its SMA(market_sma) -- i.e. the hedge appears
    exactly in the bear regime where the 2022 spread was positive, and
    is OFF in bull years where the short leg only cost money.

    Args:
        prices: DataFrame of daily close prices.
        benchmark: Series of market proxy (e.g. SPY). REQUIRED here.
        lookback: Momentum window. Defaults to 126.
        top_frac: Fraction of basket per leg. Defaults to 0.2.
        sma_filter: Per-instrument absolute trend filter. Defaults to 200.
        rebalance: Rebalance frequency. Defaults to "ME".
        skip: Days dropped at window end. Defaults to 21.
        market_sma: SMA window for the regime switch. Defaults to 200.

    Returns:
        weights: DataFrame; long-only in bull regimes, long+short in
            bear regimes.

    Notes:
        Net long in bull years (beta high), closer to neutral in bear
        years. Concentrates the short leg's value where the data says it
        exists. The honest framing for Alexander: this is a hedged
        long-only strategy, not a market-neutral one.
    """
    mom = _rank_legs(prices, lookback, top_frac, skip)

    if sma_filter is not None:
        above_sma = prices > prices.rolling(sma_filter).mean()
    else:
        above_sma = pd.DataFrame(False, index=prices.index,
                                 columns=prices.columns)

    bench = benchmark.reindex(prices.index).ffill()
    risk_off = bench < bench.rolling(market_sma).mean()

    weights = pd.DataFrame(0.0, index=prices.index,
                           columns=prices.columns)
    rebal_dates = prices.resample(rebalance).last().index

    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        row = sub.iloc[-1].dropna()
        if len(row) < 5:
            continue
        eff = sub.index[-1]

        n = max(1, int(len(row) * top_frac))
        ranked = row.sort_values(ascending=False)
        longs = list(ranked.head(n).index)

        w = pd.Series(0.0, index=prices.columns)
        if longs:
            w[longs] = 1.0 / len(longs)

        if bool(risk_off.reindex([eff]).iloc[0]):
            shorts = [s for s in ranked.tail(n).index
                      if not bool(above_sma.loc[eff, s])]
            if shorts:
                w[shorts] = -1.0 / len(shorts)

        weights.loc[dt:] = w.values

    return weights


def dual_momentum_volscaled(prices, lookback=126, top_frac=0.2,
                            sma_filter=200, rebalance="ME", skip=21,
                            vol_window=60):
    """Improvement 3: equal risk contribution per name, not equal dollars.

    Same long-top / short-bottom construction, but within each leg the
    weight is proportional to 1/volatility rather than equal dollars.
    A jumpy name (Tesla, Nvidia) gets a smaller weight than a calm one
    (KO, PG), so no single volatile name dominates the leg's risk. Legs
    are still normalised to +1 / -1 gross -> cleaner neutrality.

    Args:
        prices: DataFrame of daily close prices.
        lookback: Momentum window. Defaults to 126.
        top_frac: Fraction of basket per leg. Defaults to 0.2.
        sma_filter: Per-instrument absolute trend filter. Defaults to 200.
        rebalance: Rebalance frequency. Defaults to "ME".
        skip: Days dropped at window end. Defaults to 21.
        vol_window: Window for the inverse-vol weighting. Defaults to 60.

    Returns:
        weights: DataFrame; inverse-vol weighted within each leg, each
            leg normalised to unit gross.
    """
    mom = _rank_legs(prices, lookback, top_frac, skip)
    inv_vol = 1.0 / (prices.pct_change().rolling(vol_window).std()
                     * (252 ** 0.5))

    if sma_filter is not None:
        above_sma = prices > prices.rolling(sma_filter).mean()
    else:
        above_sma = pd.DataFrame(False, index=prices.index,
                                 columns=prices.columns)

    weights = pd.DataFrame(0.0, index=prices.index,
                           columns=prices.columns)
    rebal_dates = prices.resample(rebalance).last().index

    for dt in rebal_dates:
        sub = mom.loc[:dt]
        if sub.empty:
            continue
        row = sub.iloc[-1].dropna()
        if len(row) < 5:
            continue
        eff = sub.index[-1]

        n = max(1, int(len(row) * top_frac))
        ranked = row.sort_values(ascending=False)
        longs = list(ranked.head(n).index)
        shorts = [s for s in ranked.tail(n).index
                  if not bool(above_sma.loc[eff, s])]

        iv = inv_vol.loc[eff].replace([float("inf")], 0.0).fillna(0.0)
        w = pd.Series(0.0, index=prices.columns)

        if longs:
            lw = iv[longs]
            lw = lw / lw.sum() if lw.sum() > 0 else lw
            w[longs] = lw.values
        if shorts:
            sw = iv[shorts]
            sw = sw / sw.sum() if sw.sum() > 0 else sw
            w[shorts] = -sw.values

        weights.loc[dt:] = w.values

    return weights