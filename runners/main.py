import pandas as pd
from core.display import print_header, print_row
from core.engine import load_data, run_engine, combine_positions

from strategies.strategies import (ema_cross, ema_cross_stop, sma_trend, ema_trend,
                        ema_ensemble, ema_ensemble_voltarget,
                        ema_ensemble_long_short, ema_ensemble_voltarget_ls,
                        ema_ensemble_voltarget_smart_ls)
from strategies.strategies_turtle import (donchian_breakout,
                               donchian_ensemble_voltarget,
                               donchian_ensemble_macd_voltarget,
                               donchian_ensemble_macd_pyramid,
                               donchian_ensemble_pyramid)
from strategies.strategies_seasonal import seasonal_gas
from strategies.strategies_meanrev import bollinger_rsi
from strategies.strategies_dualmom import dual_momentum_tilt


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

minerals = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Zinc": "ZN=F",
    "Aluminum": "ALI=F",
}

energy = {
    "Crude Oil": "CL=F",
    "Brent Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Heating Oil": "HO=F",
    "Gasoline": "RB=F",
}

agriculture = {
    "Wheat": "ZW=F",
    "Corn": "ZC=F",
    "Soybeans": "ZS=F",
    "Soybean Oil": "ZL=F",
    "Soybean Meal": "ZM=F",
    "Coffee": "KC=F",
    "Cocoa": "CC=F",
    "Sugar": "SB=F",
    "Cotton": "CT=F",
}


instruments = {
    "minerals": minerals,
    "energy": energy,
    "agriculture": agriculture,
}

diversified = {
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
    # --- 10 ДОБАВЛЕННЫХ НОВЫХ АКЦИЙ ---
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
}

def run_year(trade_start, end):
    load_start = (
        pd.Timestamp(trade_start) - pd.DateOffset(years=3)
    ).strftime("%Y-%m-%d")

    print(f"\n=== Trade period: {trade_start} -> {end} ===")
    print(f"{'Instrument':<14}{'Return':>10}{'Drawdown':>10}")
    for name, ticker in minerals.items():
        close = load_data(ticker, load_start, end)
        position = ema_ensemble_voltarget(close)
        _, total_return, max_dd = run_engine(
            close, position, trade_start, cost)
        print(f"{name:<14}{total_return * 100:>9.1f}%{max_dd * 100:>9.1f}%")


"""for trade_start, end in periods:
    run_year(trade_start, end)
"""


def compare_strategies(ticker, name):
    for trade_start, end in periods:
        load_start = (
            pd.Timestamp(trade_start) - pd.DateOffset(years=3)
        ).strftime("%Y-%m-%d")
        close = load_data(ticker, load_start, end)

        print(f"\n=== {name}, {trade_start} -> {end} ===")
        print(f"{'Strategy':<14}{'Return':>10}{'Drawdown':>10}")

        for label, pos in [
            ("EMA cross", ema_cross(close)),
            # ("EMA cross stop", ema_cross_stop(close)),
            # ("SMA", sma_trend(close)),
            # ("EMA trend", ema_trend(close)),
            # ("Ensemble", ema_ensemble(close)),
            ("Ens+VolTgt", ema_ensemble_voltarget(close)),
            ("BB + RSI", bollinger_rsi(close)),
            # ("Ens+VolTgt LS", ema_ensemble_voltarget_ls(close)),
            ("Donchian", donchian_breakout(close)),
            # ("Donchian Vol", donchian_ensemble_voltarget(close)),
            # ("Donchian MACD Vol", donchian_ensemble_macd_voltarget(close)),
            # ("Donchian MACD Pyr", donchian_ensemble_macd_pyramid(close)),
            # ("Donchian Pyr", donchian_ensemble_pyramid(close)),
            # ("Donchian Pyr+S", donchian_ensemble_pyramid_seasonal(close)),
            ("Seasonal", seasonal_gas(close)),
            ("50/50 combo", combine_positions(
                donchian_breakout(close),
                bollinger_rsi(close))),
        ]:
            _, r, dd = run_engine(close, pos, trade_start, cost)
            print_row(label, r, dd)





"""
compare_strategies("GC=F", "Gold")
compare_strategies("NG=F", "Natural Gas")
compare_strategies("ALI=F", "Aluminum")
compare_strategies("NG=F", "Natural Gas")
compare_strategies("ZW=F", "Wheat")
compare_strategies("ZS=F", "Soybeans") 
compare_strategies("CC=F", "Cocoa")
compare_strategies("SB=F", "Sugar")
compare_strategies("CT=F", "Cotton")"""


"""strategies = {
    "EMA Cross": ema_cross,
    "EMA Ens+VT": ema_ensemble_voltarget,
    "Donchian MACD Vol": donchian_ensemble_macd_voltarget,
    "Donchian MACD Pyr": donchian_ensemble_macd_pyramid,
    "Dual momentum tilt": dual_momentum_tilt,
}
"""

strategies = {
    "EMA ens + vt ls": ema_ensemble_voltarget_ls,
    "EMA ens + vt ls smart": ema_ensemble_voltarget_smart_ls,
    "Donchian MACD Vol": donchian_ensemble_voltarget,
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
