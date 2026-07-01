import numpy as np
import pandas as pd

from core.engine import run_spread_engine
from core.load_panels import load_panels
# ИМПОРТИРУЕМ НОВУЮ ФУНКЦИЮ ДЛЯ СПРЕДОВ
from strategies.strategies_ou import ou_spread_zscore


TRADE_START = "2020-07-01"
N_SHUFFLE = 20            # seeds for the shuffle control
COSTS = [0.0, 0.0005, 0.001, 0.002]
BASE_COST = 0.001


def run_real(close, cost):
    # ИСПОЛЬЗУЕМ НОВЫЙ КАЛИБРАТОР
    pos = ou_spread_zscore(close)
    _, ret, dd = run_spread_engine(close, pos, TRADE_START, cost)
    return ret, dd


def run_shuffled(close, cost, seed):
    """Rebuild a random-walk price with the same price DIFFERENCES, shuffled order."""
    rng = np.random.default_rng(seed)

    # 1. ПЕРЕХОД НА АБСОЛЮТНЫЕ ПРИРАЩЕНИЯ (DIFF)
    diffs = close.diff().dropna().values
    shuffled = rng.permutation(diffs)

    synth = np.empty(len(shuffled) + 1)
    synth[0] = close.dropna().iloc[0]

    # 2. АРИФМЕТИЧЕСКОЕ БЛУЖДАНИЕ (CUMSUM)
    synth[1:] = synth[0] + np.cumsum(shuffled)

    s = pd.Series(synth, index=close.dropna().index[:len(synth)])

    # ИСПОЛЬЗУЕМ НОВЫЙ КАЛИБРАТОР
    pos = ou_spread_zscore(s)
    _, ret, dd = run_spread_engine(s, pos, TRADE_START, cost)
    return ret


def main():
    print("Загрузка данных...")
    panels = load_panels()
    
    # === Бронебойный парсер данных ===
    close_prices = {}
    
    # Шаг 1: Извлекаем таблицу (DataFrame) из объекта
    if isinstance(panels, dict):
        if "Close" in panels:
            df = panels["Close"]
        else:
            # Если ключа Close нет, просто берем первую попавшуюся таблицу из словаря
            df = list(panels.values())[0]
    else:
        # Если это уже DataFrame
        df = panels

    # Шаг 2: Разбиваем таблицу на отдельные Series (каждая колонка = 1 тикер)
    if isinstance(df, pd.DataFrame):
        for sym in df.columns:
            close_prices[sym] = df[sym]
    elif isinstance(df, pd.Series):
        close_prices[df.name or "Asset"] = df
    else:
        print("Ошибка: Не удалось найти DataFrame с ценами.")
        return

    print(f"\n{'-'*70}")
    print(f"{'SYM':<5}{'Real PnL':>12}{'Med Shuf':>12}{'90% Shuf':>12}{'Edge':>9}{'Max DD':>12}")
    print("-" * 70)
    
    verdicts = {}
    
    # Итерируемся строго по одному инструменту (Series)
    for sym, px in close_prices.items():
        px = px.dropna()
        if len(px) < 100: # Пропускаем пустые данные
            continue
            
        # Запускаем прогон на реальных ценах
        rr, dd = run_real(px, BASE_COST)
        
        # Запускаем случайные блуждания
        shuffles = [run_shuffled(px, BASE_COST, i) for i in range(N_SHUFFLE)]
        med = np.median(shuffles)
        p90 = np.percentile(shuffles, 90)
        
        edge = "YES" if rr > p90 else "no"
        verdicts[sym] = edge
        
        print(f"{sym:<5}{rr:>12.2f}{med:>12.2f}{p90:>12.2f}{edge:>9}{dd:>12.2f}")
        
    print("-" * 70)
    n_yes = sum(1 for v in verdicts.values() if v == "YES")
    print(f"  edge=YES (Real > 90th percentile of shuffles): {n_yes} / {len(verdicts)}")

if __name__ == "__main__":
    main()
