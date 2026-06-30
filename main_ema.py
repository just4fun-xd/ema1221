import pandas as pd
from display import print_row
from engine import load_data, run_engine

from strategies import (ema_cross, ema_cross_stop, sma_trend, ema_trend,
                        ema_ensemble, ema_ensemble_voltarget,
                        ema_ensemble_long_short, ema_ensemble_voltarget_ls)


cost = 0.001

periods = [
    ("2019-01-01", "2020-01-01"),
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]


diversified = {
    # Commodities
    "Gold": "GC=F",
    "Palladium": "PA=F",
    "Zinc": "ZN=F",
    "Aluminum": "ALI=F",
    "Crude Oil": "CL=F",
    "Copper": "HG=F",
    "Brent Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Heating Oil": "HO=F",
    "Gasoline": "RB=F",
    "Wheat": "ZW=F",
    "Corn": "ZC=F",
    "Soybeans": "ZS=F",
    "Soybean Oil": "ZL=F",
    "Soybean Meal": "ZM=F",
    "Coffee": "KC=F",
    "Cocoa": "CC=F",
    "Sugar": "SB=F",
    "Cotton": "CT=F",
    # Equity indices (ETF — надёжно, без roll)
    "S&P 500": "SPY",
    "Nasdaq": "QQQ",

    # Equity (Tech & Growth)
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Meta": "META",
    "Alphabet": "GOOGL",
    "AMD": "AMD",
    "Netflix": "NFLX",
    "Visa": "V",
    "Mastercard": "MA",
    "UnitedHealth": "UNH",
    "Home Depot": "HD",
    "Bank of America": "BAC",
    "Coca-Cola": "KO",
    "Procter & Gamble": "PG",
    "Walmart": "WMT",
    # ------------------------------------

    # Equity (Other sectors)
    "JPMorgan": "JPM",
    "Exxon Mobil": "XOM",
    "Eli Lilly": "LLY",
    # Bonds (ETF)
    "20Y Treasury": "TLT",
    "7-10Y Treasury": "IEF",
    # Currencies (фьючерс)
    "Euro": "6E=F",
    "Yen": "6J=F",
    # Crypto (короткая история, но сильные тренды)
    # "Bitcoin": "BTC-USD",
}

strategies = {
    "EMA Cross": ema_cross,
    "EMA Cross+Stop": ema_cross_stop,
    "SMA Trend": sma_trend,
    "EMA Trend": ema_trend,
    "EMA Ens": ema_ensemble,
    "EMA Ens+VT": ema_ensemble_voltarget,
    "EMA Ens LS": ema_ensemble_long_short,
    "EMA Ens+VT LS": ema_ensemble_voltarget_ls,
}


def run_year_new(trade_start, end, instr, group_name, strategy_name, strategy_func):
    load_start = (
        pd.Timestamp(trade_start) - pd.DateOffset(years=3)
    ).strftime("%Y-%m-%d")

    print(f"\n{'='*50}")
    print(f"=== Group: {group_name} ===")
    print(f"=== Strategy: {strategy_name} ===")
    print(f"=== Period: {trade_start} -> {end} ===")
    print(f"{'Instrument':<14}{'Return':>10}{'Drawdown':>10}")
    print("-" * 34)

    for name, ticker in instr.items():
        close = load_data(ticker, load_start, end)

        position = strategy_func(close)

        _, total_return, max_dd = run_engine(close, position, trade_start, cost)
        print_row(name, total_return, max_dd)


instruments_div = {
    "Diversified Portfolio": diversified
}


for trade_start, end in periods:
    for group_name, group_dict in instruments_div.items():
        for strategy_name, strategy_func in strategies.items():
            run_year_new(
                trade_start,
                end,
                group_dict,
                group_name,
                strategy_name,
                strategy_func
            )
