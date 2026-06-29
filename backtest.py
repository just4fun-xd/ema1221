import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


trade_start = "2025-06-24"
end = "2026-06-24"
cost = 0.001

periods = [
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

instruments = {
    "Gold": "GC=F",           # Золото
    "Silver": "SI=F",         # Серебро
    "Copper": "HG=F",         # Медь
    "Crude Oil": "CL=F",      # Нефть WTI
    "Natural Gas": "NG=F",    # Природный газ
    "Corn": "ZC=F",           # Кукуруза
    "Wheat": "ZW=F",          # Пшеница
    "Soybeans": "ZS=F",       # Соевые бобы
}


def load_data(ticker, load_start, end):
    data = yf.download(ticker, start=load_start, end=end,
                       interval="1d", progress=False)
    data.columns = data.columns.droplevel(1)
    return data["Close"]


def run_backtest(close, trade_start, cost=0.001, filter_period=200):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    result = pd.DataFrame({
        "Close": close,
        "EMA12": ema12,
        "EMA21": ema21
    })

    cross = result["EMA12"] > result["EMA21"]

    if filter_period is None:
        result["bull"] = cross
    else:
        ema_filter = close.ewm(span=filter_period, adjust=False).mean()
        result["EMAfilter"] = ema_filter
        uptrend = result["Close"] > result["EMAfilter"]
        result["bull"] = cross & uptrend

    result["trade"] = result["bull"] != result["bull"].shift(1)
    result["returns"] = result["Close"] / result["Close"].shift(1) - 1
    result["strategy"] = result["returns"] * result["bull"].shift(1)
    result["strategy"] = result["strategy"] - result["trade"] * cost

    result = result[result.index >= trade_start]

    result["equity"] = (1 + result["strategy"]).cumprod()

    result["peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["peak"] - 1

    total_return = result["equity"].iloc[-1] - 1
    max_dd = result["drawdown"].min()
    return result, total_return, max_dd


def plot_result(result, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(result.index, result["Close"],
             label=name, color="black", linewidth=1)
    ax1.plot(result.index, result["EMA12"], label="EMA12", color="blue")
    ax1.plot(result.index, result["EMA21"], label="EMA21", color="red")
    ax1.plot(result.index, result["EMAfilter"],
             label="EMAfilter", color="orange", linewidth=1)

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(result.index, result["equity"], label="Equity", color="green")
    ax2.set_title("Strategy equity curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


results = {}


def run_year(trade_start, end):
    load_start = (
        pd.Timestamp(trade_start) - pd.DateOffset(years=3)
    ).strftime("%Y-%m-%d")

    print(f"\n=== Trade period: {trade_start} -> {end} ===")
    print(f"{'Instrument':<14}{'Return':>10}{'Drawdown':>10}")
    for name, ticker in instruments.items():
        close = load_data(ticker, load_start, end)
        _, total_return, max_dd = run_backtest(
            close, trade_start, cost, filter_period=None)
        print(f"{name:<14}{total_return * 100:>9.1f}%{max_dd * 100:>9.1f}%")


for trade_start, end in periods:
    run_year(trade_start, end)
