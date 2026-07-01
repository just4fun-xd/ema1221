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
