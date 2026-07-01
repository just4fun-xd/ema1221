import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.engine import run_spread_engine
from core.load_panels import load_panels
from strategies.strategies_ou import ou_spread_zscore

TRADE_START = "2020-07-01"
N_SHUFFLE = 20            
# Стоимость чуть выше, так как мы платим комиссию за 2 ноги (покупка актива А и продажа актива B)
BASE_COST = 0.02  

def run_real(spread_series, cost):
    pos = ou_spread_zscore(spread_series)
    _, ret, dd = run_spread_engine(spread_series, pos, TRADE_START, cost)
    return ret, dd

def run_shuffled(spread_series, cost, seed):
    rng = np.random.default_rng(seed)
    diffs = spread_series.diff().dropna().values
    shuffled = rng.permutation(diffs)
    
    synth = np.empty(len(shuffled) + 1)
    synth[0] = spread_series.dropna().iloc[0]
    synth[1:] = synth[0] + np.cumsum(shuffled)
    
    s = pd.Series(synth, index=spread_series.dropna().index[:len(synth)])
    pos = ou_spread_zscore(s)
    _, ret, dd = run_spread_engine(s, pos, TRADE_START, cost)
    return ret

def get_hedge_ratio(y, x):
    """
    Вычисляет коэффициент хеджирования (бета) на обучающей выборке (первые 252 дня).
    Это исключает 'заглядывание в будущее' (look-ahead bias) при бэктесте.
    """
    train_y = y.iloc[:252].dropna()
    train_x = x.iloc[:252].dropna()
    
    df = pd.concat([train_y, train_x], axis=1).dropna()
    if len(df) < 100:
        return 1.0
        
    # OLS регрессия: Y = a + Beta * X
    model = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit()
    return model.params.iloc[1] # Возвращаем Beta

def main():
    print("Загрузка панелей данных...")
    panels = load_panels()
    close = panels["close"] # Используем правильный ключ из твоего загрузчика
    
    # Классические коинтегрированные пары для стат. арбитража
    pairs = [
        ("GC", "SI"),  # Золото - Серебро (Драгметаллы)
        ("CL", "HO"),  # Нефть - Мазут (Энергетика / Crack Spread)
        ("ZW", "ZC"),  # Пшеница - Кукуруза (Зерновые)
        ("ZS", "ZM"),  # Соя - Соевая мука (Crush Spread)
        ("HG", "SI"),  # Медь - Серебро (Промышленные vs Драгметаллы)
    ]

    print(f"\n{'-'*75}")
    print(f"{'PAIR':<10}{'Beta':>8}{'Real PnL':>12}{'90% Shuf':>12}{'Edge':>9}{'Max DD':>12}")
    print("-" * 75)
    
    verdicts = {}

    for y_sym, x_sym in pairs:
        if y_sym not in close.columns or x_sym not in close.columns:
            print(f"Пропуск {y_sym}-{x_sym} (нет данных)")
            continue
            
        y = close[y_sym].dropna()
        x = close[x_sym].dropna()
        
        # Строго синхронизируем индексы по датам (чтобы вычитать день в день)
        df = pd.concat([y, x], axis=1).dropna()
        y_sync = df.iloc[:, 0]
        x_sync = df.iloc[:, 1]
        
        # 1. Находим множитель Beta
        beta = get_hedge_ratio(y_sync, x_sync)
        
        # 2. Строим стационарный синтетический спред!
        spread = y_sync - beta * x_sync
        
        # 3. Тестируем на Грааль
        rr, dd = run_real(spread, BASE_COST)
        shuffles = [run_shuffled(spread, BASE_COST, i) for i in range(N_SHUFFLE)]
        p90 = np.percentile(shuffles, 90)
        
        edge = "YES" if rr > p90 else "no"
        verdicts[f"{y_sym}-{x_sym}"] = edge
        
        print(f"{y_sym}-{x_sym:<7}{beta:>8.2f}{rr:>12.2f}{p90:>12.2f}{edge:>9}{dd:>12.2f}")

    print("-" * 75)
    n_yes = sum(1 for v in verdicts.values() if v == "YES")
    print(f"  Pairs with Edge=YES: {n_yes} / {len(verdicts)}")

if __name__ == "__main__":
    main()
