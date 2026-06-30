import pandas as pd
import numpy as np

from core.engine import load_data, run_engine


def load_basket(tickers, load_start, end):
    """Load a basket of tickers into one aligned price DataFrame.

    Reuses the project's load_data (yfinance) per ticker, then aligns
    everything on a common date index. Tickers that fail to load are
    skipped with a warning rather than killing the whole run.

    Args:
        tickers: Dict of {name: ticker} or list of ticker strings.
        load_start: Start date string passed to load_data.
        end: End date string passed to load_data.

    Returns:
        prices: DataFrame of close prices (columns = names/tickers).
    """
    if isinstance(tickers, dict):
        items = list(tickers.items())
    else:
        items = [(t, t) for t in tickers]

    cols = {}
    for name, ticker in items:
        try:
            s = load_data(ticker, load_start, end)
            if s is not None and len(s) > 0:
                cols[name] = s
        except Exception as e:
            print(f"  skip {name} ({ticker}): {e}")

    prices = pd.DataFrame(cols).sort_index()
    return prices


def positions_to_weights(prices, strategy_fn, gross=1.0, **kwargs):
    """Wrap a per-series strategy into a portfolio weight matrix.

    Runs the existing single-instrument strategy on each column, then
    normalises the raw positions each day so the basket has constant
    gross exposure. This lets old strategies (ema_cross, donchian_*)
    run through run_portfolio for an apples-to-apples comparison with
    cross-sectional ones -- WITHOUT rewriting them.

    Args:
        prices: DataFrame of close prices (columns = instruments).
        strategy_fn: Any existing strategy taking close: Series ->
            position: Series.
        gross: Target gross exposure summed across instruments each day.
            Defaults to 1.0.
        **kwargs: Passed through to strategy_fn.

    Returns:
        weights: DataFrame of weights, same grid as prices.

    Notes:
        This is an adapter, not a new strategy. The per-series logic and
        its shift(1) discipline are unchanged -- run_portfolio applies
        shift(1) exactly once, same as run_engine.
    """
    raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        raw[col] = strategy_fn(prices[col], **kwargs)

    gross_row = raw.abs().sum(axis=1).replace(0, np.nan)
    weights = raw.div(gross_row, axis=0).fillna(0.0) * gross
    return weights


def run_portfolio(prices, weights, trade_start, cost=0.001,
                  target_vol=None, vol_window=30, max_leverage=2.0):
    """Backtest a weight matrix on a basket of instruments.

    Portfolio analogue of run_engine. Same contract: shift(1) lives in
    exactly one place (here), so strategies return raw target weights and
    cannot peek at today's return. Costs are charged on turnover.

    Args:
        prices: DataFrame of daily close prices (columns = instruments).
        weights: DataFrame of target weights, same grid as prices.
        trade_start: Date string; equity is measured from here.
        cost: Per-unit-turnover transaction cost (0.001 = 10 bps).
        target_vol: If set, scale the whole portfolio so realised
            annual vol matches this. None disables it. Defaults to None.
        vol_window: Rolling window for the vol estimate. Defaults to 30.
        max_leverage: Cap on the vol-targeting multiplier. Defaults to 2.0.

    Returns:
        result: DataFrame with port_return, equity, drawdown.
        total_return: Final cumulative return.
        max_dd: Worst drawdown over the period.
    """
    prices = prices.sort_index()
    weights = weights.reindex(prices.index).fillna(0.0)

    asset_ret = prices.pct_change()
    held = weights.shift(1)

    if target_vol is not None:
        gross_ret = (asset_ret * held).sum(axis=1)
        realised = gross_ret.rolling(vol_window).std() * np.sqrt(252)
        scale = (target_vol / realised).clip(upper=max_leverage)
        scale = scale.shift(1).fillna(0.0)
        held = held.mul(scale, axis=0)

    port_ret = (asset_ret * held).sum(axis=1)

    turnover = (held - held.shift(1)).abs().sum(axis=1)
    port_ret = port_ret - turnover * cost

    result = pd.DataFrame({"port_return": port_ret})
    result = result[result.index >= trade_start]

    result["equity"] = (1 + result["port_return"]).cumprod()
    result["peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["peak"] - 1

    total_return = result["equity"].iloc[-1] - 1
    max_dd = result["drawdown"].min()
    return result, total_return, max_dd


def portfolio_beta(prices, weights, benchmark, trade_start):
    """Beta of the weighted portfolio to a benchmark series.

    Test #1 of the honesty checklist from SHORT_REDESIGN.md: a
    market-neutral construction should have beta close to zero.

    Args:
        prices: DataFrame of daily close prices.
        weights: DataFrame of target weights.
        benchmark: Series of benchmark close prices (e.g. SPY).
        trade_start: Date string; beta measured from here.

    Returns:
        beta: float, slope of portfolio returns on benchmark returns.
    """
    asset_ret = prices.pct_change()
    held = weights.reindex(prices.index).fillna(0.0).shift(1)
    port_ret = (asset_ret * held).sum(axis=1)

    bench_ret = benchmark.pct_change().reindex(port_ret.index)

    df = pd.DataFrame({"p": port_ret, "b": bench_ret})
    df = df[df.index >= trade_start].dropna()
    cov = df["p"].cov(df["b"])
    var = df["b"].var()
    return cov / var if var else float("nan")


def run_portfolio_yearly(prices, weights, periods, cost=0.001,
                         target_vol=None, vol_window=30,
                         max_leverage=2.0):
    """Per-instrument, per-year breakdown -- old dev_test.py style report.

    For each (trade_start, end) period, isolates each instrument's
    contribution by running a single-column portfolio (that instrument's
    weight column only) through run_portfolio. Prints a Return / Drawdown
    table per period, matching the project's familiar output format.

    Args:
        prices: DataFrame of close prices (columns = instruments).
        weights: DataFrame of target weights, same grid as prices.
        periods: List of (trade_start, end) date-string tuples.
        cost: Per-unit-turnover cost. Defaults to 0.001.
        target_vol: Optional per-instrument vol target. Defaults to None.
        vol_window: Vol estimate window. Defaults to 30.
        max_leverage: Vol-targeting cap. Defaults to 2.0.

    Returns:
        None. Prints the report.

    Notes:
        This shows each instrument's standalone P&L given its weight in
        the matrix -- NOT its marginal contribution to the joint
        portfolio (legs interact). Use it to see WHERE longs/shorts land,
        the same diagnostic value as the old per-series report.
    """
    try:
        from core.display import print_row
    except Exception:
        def print_row(label, r, dd, col_width=14):
            print(f"{label:<{col_width}}{r*100:>9.1f}%{dd*100:>9.1f}%")

    for trade_start, end in periods:
        win = prices[(prices.index >= trade_start) & (prices.index < end)]
        if win.empty:
            continue
        print(f"\n=== Period: {trade_start} -> {end} ===")
        print(f"{'Instrument':<18}{'Return':>10}{'Drawdown':>10}")
        print("-" * 38)
        for col in prices.columns:
            w_col = weights[[col]].copy()
            p_col = prices[[col]]
            sub_p = p_col[p_col.index < end]
            sub_w = w_col[w_col.index < end]
            try:
                _, r, dd = run_portfolio(
                    sub_p, sub_w, trade_start, cost,
                    target_vol=target_vol, vol_window=vol_window,
                    max_leverage=max_leverage)
                print_row(col, r, dd, col_width=18)
            except Exception as e:
                print(f"{col:<18}  ERROR: {e}")