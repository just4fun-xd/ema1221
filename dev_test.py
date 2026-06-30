"""
Диверсифицированный бэктест: проверка тезиса Александра
"trend following can work, just not in the case you programmed it for".

Идея: применить donchian_ensemble_pyramid и ema_ensemble_voltarget
к ДИВЕРСИФИЦИРОВАННОЙ корзине (не только commodities) на ДЛИННОМ
горизонте (2010-2025), с ETF вместо фьючерсов где можно
(нет roll-проблемы — решает Проблему 5 из разгрома).

Запускать локально: python diversified_test.py
"""

import pandas as pd
from engine import load_data, run_engine
from strategies import (
     ema_cross, ema_cross_stop, sma_trend, ema_trend,
     ema_ensemble, ema_ensemble_voltarget,
     ema_ensemble_long_short, ema_ensemble_voltarget_ls)

from strategies_turtle import (donchian_ensemble_pyramid, donchian_breakout,
                               donchian_breakout_ls,
                               donchian_ensemble_voltarget,
                               )
from not_now import (donchian_ensemble_macd_4step_pyramid,
                     donchian_ensemble_macd_4step_take)

from strategies_seasonal import (seasonal_gas, donchian_seasonal,
                                 donchian_seasonal_voltarget)
from strategies_bollinger import donchian_bollinger_b, bollinger_squeeze

cost = 0.001

# ── Диверсифицированная корзина: 5 классов активов ────────────────────────
# ETF где возможно (без roll-проблемы), фьючерсы для commodities
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

"""diversified = {
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
}"""

"""diversified = {
    # Equity indices (ETF — надёжно, без roll)
    "S&P 500": "SPY",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Meta": "META",
    "Alphabet": "GOOGL",
    "JPMorgan": "JPM",
    "Exxon Mobil": "XOM",
    "Eli Lilly": "LLY",
    "Nasdaq": "QQQ",
    # Bonds (ETF)
    "20Y Treasury": "TLT",
    "7-10Y Treasury": "IEF",
    # Currencies (фьючерс)
    "Euro": "6E=F",
    "Yen": "6J=F",
    # Crypto (короткая история, но сильные тренды)
    # "Bitcoin": "BTC-USD",
}"""

# ── Длинный горизонт ──────────────────────────────────────────────────────
periods = [
    # ("2013-01-01", "2014-01-01"),
    # ("2014-01-01", "2015-01-01"),
    # ("2015-01-01", "2016-01-01"),
    # ("2016-01-01", "2017-01-01"),
    # ("2017-01-01", "2018-01-01"),
    # ("2018-01-01", "2019-01-01"),
    # ("2019-01-01", "2020-01-01"),
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

"""strategies = {
    "Donchian Pyr": donchian_ensemble_pyramid,
    "EMA Ens+VT": ema_ensemble_voltarget,
    "Donchian ": donchian_breakout,
    "Donchian LS": donchian_breakout_ls,
    "Donchian Est+VT": donchian_ensemble_voltarget,
    "Don Est+macd+4step+pyr": donchian_ensemble_macd_4step_pyramid,
    "Don Est+macd+4step+take": donchian_ensemble_macd_4step_take,
    "Don bollinger b": donchian_bollinger_b,
}"""

"""strategies = {
    "Seasonal gas": seasonal_gas,
    "Donchian Seasonal": donchian_seasonal,
    "Donchian Seasonal + VT ": donchian_seasonal_voltarget,

}"""


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


def run_diversified(strategy_name, strategy_fn):
    """Прогон одной стратегии по всей корзине, итог $100 на инструмент."""
    print(f"\n{'='*70}")
    print(f"  {strategy_name} — diversified basket, $100 per instrument")
    print(f"{'='*70}")
    print(f"{'Instrument':<16}{'Total Ret':>12}{'Max DD':>10}{'Final $':>12}")
    print("-" * 70)

    finals = []
    all_dd = []
    for name, ticker in diversified.items():
        capital = 100.0
        worst_dd = 0.0
        try:
            for trade_start, end in periods:
                load_start = (
                    pd.Timestamp(trade_start) - pd.DateOffset(years=3)
                ).strftime("%Y-%m-%d")
                close = load_data(ticker, load_start, end)
                if close is None or len(close) < 250:
                    continue
                position = strategy_fn(close)
                _, ret, dd = run_engine(close, position, trade_start, cost)
                capital *= (1 + ret)
                worst_dd = min(worst_dd, dd)
        except Exception as e:
            print(f"{name:<16}  ERROR: {e}")
            continue

        finals.append(capital)
        all_dd.append(worst_dd)
        total_ret = (capital / 100 - 1) * 100
        print(f"{name:<16}{total_ret:>11.1f}%{worst_dd*100:>9.1f}%"
              f"${capital:>11.2f}")

    print("-" * 70)
    if finals:
        port_final = sum(finals) / len(finals)
        port_ret = (port_final / 100 - 1) * 100
        worst = min(all_dd) * 100
        print(f"{'PORTFOLIO (avg)':<16}{port_ret:>11.1f}%{worst:>9.1f}%"
              f"${port_final:>11.2f}")
        n_profit = sum(1 for f in finals if f > 100)
        print(f"\nProfitable instruments: {n_profit}/{len(finals)}")


if __name__ == "__main__":
    for sname, sfn in strategies.items():
        run_diversified(sname, sfn)

    print("\n" + "="*70)
    print("  Интерпретация:")
    print("  - Если портфель на диверсифицированной корзине лучше, чем")
    print("    на чистых commodities → тезис Александра подтверждён.")
    print("  - Bitcoin может сильно тянуть вверх (мощные тренды) — смотри")
    print("    и с ним, и без него (исключи строку из basket).")
    print("  - ETF (SPY/QQQ/TLT/IEF) не имеют roll-проблемы — честнее.")
    print("="*70)
