"""S&P 500 universe loader for wide-basket cross-sectional tests.

Fetches the CURRENT S&P 500 constituents from Wikipedia and caches them
to a local CSV. Used to test whether dual_momentum's concentration on a
handful of mega-caps (Nvidia/Tesla/AMD = 66% of P&L on the 19-name
basket) dissolves when the basket is wide.

HONEST LIMITATION -- survivorship bias:
    Wikipedia lists only TODAY's members. Companies that were dropped
    from the index (bankruptcies, mergers, underperformers) are absent.
    So this basket is biased toward survivors -- it overstates returns
    for ANY long strategy. This does NOT invalidate the concentration
    test (we are asking whether the edge spreads across names, not what
    the absolute return is), but the absolute numbers must be reported
    with this caveat. A true point-in-time universe needs a historical
    constituents source (e.g. bkestelman/sp500_historical_components),
    out of scope for the yfinance stack.
"""

import os
import pandas as pd


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE = "sp500_tickers.csv"


def get_sp500_tickers(limit=None, use_cache=True):
    """Return current S&P 500 tickers, yfinance-formatted.

    Args:
        limit: If set, return only the first N tickers (faster testing).
        use_cache: Read/write a local CSV cache to avoid re-scraping.

    Returns:
        dict of {ticker: ticker} suitable for load_basket. (Name == ticker;
        the company name column is kept in the cache CSV if you want it.)

    Notes:
        Tickers like BRK.B / BF.B come from Wikipedia with a dot; yfinance
        wants a hyphen (BRK-B). We convert here -- without this they fail
        to download and silently drop out of the basket.
    """
    if use_cache and os.path.exists(CACHE):
        df = pd.read_csv(CACHE)
    else:
        import io
        import requests
        # Wikipedia returns 403 to bare pd.read_html (no User-Agent).
        # Fetch with a browser-like header first, then parse the HTML.
        headers = {"User-Agent": "Mozilla/5.0 (research script)"}
        resp = requests.get(WIKI_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["ticker", "name", "sector"]
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        if use_cache:
            df.to_csv(CACHE, index=False)

    tickers = df["ticker"].tolist()
    if limit:
        tickers = tickers[:limit]
    return {t: t for t in tickers}


def load_basket_batch(tickers, load_start, end, min_rows=250):
    """Download many tickers in ONE yfinance call (fast), then clean.

    Per-ticker download (load_basket) is fine for 19 names but slow and
    rate-limit-prone for 500. yfinance.download accepts a list and returns
    a multi-index frame; we slice the Close level and drop names with too
    little history (recent listings, failed downloads).

    Args:
        tickers: dict {name: ticker} or list of ticker strings.
        load_start: start date string.
        end: end date string.
        min_rows: drop any column with fewer than this many valid closes.

    Returns:
        prices: DataFrame of Close prices (columns = tickers), cleaned.

    Notes:
        Survivorship bias applies (see module docstring). Columns are
        forward-filled for small gaps but dropped entirely if they are
        mostly empty -- a half-empty column poisons cross-sectional ranks.
    """
    import yfinance as yf

    if isinstance(tickers, dict):
        syms = list(tickers.values())
    else:
        syms = list(tickers)

    raw = yf.download(syms, start=load_start, end=end, interval="1d",
                      progress=False, group_by="column", auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = syms[:1]

    close = close.dropna(axis=1, how="all")
    valid = close.notna().sum()
    keep = valid[valid >= min_rows].index
    close = close[keep].ffill()

    dropped = len(syms) - close.shape[1]
    if dropped:
        print(f"  dropped {dropped} tickers "
              f"(insufficient history / failed download)")
    return close
