import yfinance as yf
import pandas as pd


def load_data(ticker, load_start, end):
    data = yf.download(ticker, start=load_start, end=end,
                       interval="1d", progress=False)
    data.columns = data.columns.droplevel(1)
    return data["Close"]


def load_ohlc(ticker, load_start, end):
    data = yf.download(ticker, start=load_data, end=end,
                       interval="1d", progress=False)
    data.columns = data.columns.droplevel(1)
    return data[["High", "Low", "Close"]]


def run_engine(close, position, trade_start, cost=0.001):
    result = pd.DataFrame({"Close": close, "position": position})
    held = result["position"].shift(1)

    result["trade"] = (result["position"] - held).abs()

    result["returns"] = result["Close"] / result["Close"].shift(1) - 1
    result["strategy"] = result["returns"] * held
    result["strategy"] = result["strategy"] - result["trade"] * cost

    result = result[result.index >= trade_start]

    result["equity"] = (1 + result["strategy"]).cumprod()
    result["peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["peak"] - 1

    total_return = result["equity"].iloc[-1] - 1
    max_dd = result["drawdown"].min()
    return result, total_return, max_dd


def combine_positions(*positions, weights=None):
    """Average multiple strategy positions into one combined signal.

    Args:
        *positions: Any number of position Series (same index).
        weights: Optional list of weights. Defaults to equal weight.

    Returns:
        combined: Weighted average position Series.
    """
    if weights is None:
        weights = [1.0 / len(positions)] * len(positions)
    combined = sum(p * w for p, w in zip(positions, weights))
    return combined
