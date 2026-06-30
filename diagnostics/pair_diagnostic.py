import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from core.engine import load_data


def analyze_pair(name_a, ticker_a, name_b, ticker_b,
                 start="2015-01-01", end="2025-01-01"):
    """Diagnose a candidate pair for cointegration-based spread trading.

    Computes the hedge ratio (OLS), tests spread stationarity (ADF),
    and estimates mean-reversion half-life (OU via AR(1)). Prints a
    verdict — does NOT trade. Run this BEFORE writing any strategy on
    a pair: if the spread is not stationary, mean-reversion will fail.

    Args:
        name_a, ticker_a: Label and yfinance ticker of asset A.
        name_b, ticker_b: Label and yfinance ticker of asset B.
        start, end: Date range for the diagnostic window.

    Returns:
        None. Prints beta, ADF p-value, half-life, and a verdict.
    """
    a = load_data(ticker_a, start, end)
    b = load_data(ticker_b, start, end)

    # Выравниваем по общим датам (разные инструменты — разные пропуски)
    df = pd.concat([a, b], axis=1, keys=["A", "B"], sort=True).dropna()
    a = df["A"]
    b = df["B"]

    # --- Слой 1: hedge ratio через OLS (A = alpha + beta*B + eps) ---
    x = sm.add_constant(b)
    model = sm.OLS(a, x).fit()
    beta = model.params.iloc[1]
    spread = a - beta * b

    # --- Слой 2: ADF-тест на стационарность остатков ---
    adf_stat, adf_p, *_ = adfuller(spread.dropna())

    # --- Слой 3: half-life через OU / AR(1) ---
    # spread[t] - spread[t-1] = lambda * spread[t-1] + c + eps
    spread_lag = spread.shift(1).dropna()
    spread_delta = spread.diff().dropna()
    spread_lag = spread_lag.loc[spread_delta.index]
    x_hl = sm.add_constant(spread_lag)
    hl_model = sm.OLS(spread_delta, x_hl).fit()
    lam = hl_model.params.iloc[1]
    half_life = -np.log(2) / lam if lam < 0 else np.inf

    # --- Вердикт ---
    print(f"\n{'='*60}")
    print(f"  {name_a} / {name_b}")
    print(f"{'='*60}")
    print(f"  beta (hedge ratio):  {beta:.4f}")
    print(f"  ADF p-value:         {adf_p:.4f}", end="")
    print("  ✅ стационарен" if adf_p < 0.05 else "  ❌ НЕ стационарен")
    print(f"  half-life:           {half_life:.1f} дней", end="")
    if half_life == np.inf or half_life < 0:
        print("  ❌ не mean-reverting")
    elif half_life > 100:
        print("  ⚠️  слишком медленный (>100)")
    else:
        print("  ✅ разумный")

    cointegrated = adf_p < 0.05 and 0 < half_life < 100
    print(f"\n  ВЕРДИКТ: {'✅ ТОРГУЕМ' if cointegrated else '❌ ПРОПУСКАЕМ'}")


pairs = [
    # --- Драгметаллы (общий драйвер: реальные ставки, доллар, risk-off) ---
    ("Gold", "GC=F", "Silver", "SI=F"),
    ("Gold", "GC=F", "Platinum", "PL=F"),
    ("Platinum", "PL=F", "Palladium", "PA=F"),
    ("Silver", "SI=F", "Platinum", "PL=F"),

    # --- Энергия (общий драйвер: цена нефти, спрос на топливо) ---
    ("Crude Oil", "CL=F", "Brent Oil", "BZ=F"),
    ("Heating Oil", "HO=F", "Gasoline", "RB=F"),
    ("Crude Oil", "CL=F", "Heating Oil", "HO=F"),
    ("Crude Oil", "CL=F", "Gasoline", "RB=F"),
    ("Brent Oil", "BZ=F", "Heating Oil", "HO=F"),

    # --- Зерновые (общий драйвер: погода, посевные площади, спрос) ---
    ("Corn", "ZC=F", "Wheat", "ZW=F"),
    ("Corn", "ZC=F", "Soybeans", "ZS=F"),
    ("Wheat", "ZW=F", "Soybeans", "ZS=F"),

    # --- Соевый комплекс (crush spread: бобы → масло + шрот) ---
    ("Soybeans", "ZS=F", "Soybean Oil", "ZL=F"),
    ("Soybeans", "ZS=F", "Soybean Meal", "ZM=F"),
    ("Soybean Oil", "ZL=F", "Soybean Meal", "ZM=F"),

    # --- Промышленные металлы (общий драйвер: глобальный рост, Китай) ---
    ("Copper", "HG=F", "Zinc", "ZN=F"),

    # --- Softs (слабая связь, скорее контроль — должны НЕ коинтегрироваться) ---
    ("Coffee", "KC=F", "Cocoa", "CC=F"),
    ("Sugar", "SB=F", "Cotton", "CT=F"),
]


if __name__ == "__main__":
    for na, ta, nb, tb in pairs:
        analyze_pair(na, ta, nb, tb)