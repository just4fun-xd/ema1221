import numpy as np
import pandas as pd
import statsmodels.api as sm


def calibrate_ou(spread_series):
    """
    Калибрует параметры процесса Орнштейна-Уленбека (OU) для временного ряда.
    Использует линейную регрессию AR(1).
    """
    s = spread_series.dropna()
    if len(s) < 30:
        return None

    Y = s.values[1:]
    X = s.values[:-1]

    # ЗАЩИТА 1: Если цена стояла на месте (нулевая дисперсия), 
    # математика возврата к среднему не работает. Пропускаем.
    if np.var(X) == 0:
        return {'mu': np.nan, 'half_life': np.nan, 'sigma_eq': np.nan, 'is_stationary': False}

    X_with_const = sm.add_constant(X)

    # Обучаем модель OLS
    model = sm.OLS(Y, X_with_const).fit()

    # ЗАЩИТА 2: Если из-за качества данных statsmodels вернул не 2 коэффициента
    if len(model.params) != 2:
         return {'mu': np.nan, 'half_life': np.nan, 'sigma_eq': np.nan, 'is_stationary': False}

    a, b = model.params

    # Проверка на стационарность (если b >= 1, то это тренд, возврата нет)
    if b <= 0 or b >= 1:
        return {'mu': np.nan, 'half_life': np.nan, 'sigma_eq': np.nan, 'is_stationary': False}

    theta = -np.log(b)
    mu = a / (1 - b)
    half_life = np.log(2) / theta

    # Равновесная волатильность
    var_residuals = np.var(model.resid)
    sigma_eq = np.sqrt(var_residuals / (1 - b**2)) 

    return {
        'mu': mu, 
        'half_life': half_life, 
        'sigma_eq': sigma_eq, 
        'is_stationary': True
    }


def ou_spread_zscore(spread, lookback=90, entry=2.0, exit=0.5, stop=4.0, max_hold=None):
    """
    Настоящий Орнштейн-Уленбек для календарных спредов.
    Вместо скользящей средней использует динамическую калибровку параметров OU
    за последние `lookback` дней.
    """
    pos = pd.Series(0.0, index=spread.index)
    state = 0.0          
    hold = 0             

    sv = spread.values
    out = np.zeros(len(sv))

    for i in range(len(sv)):
        # Ждем накопления истории для первой калибровки
        if i < lookback:
            out[i] = 0.0
            continue

        # Берем окно данных до текущего дня включительно
        window = spread.iloc[i - lookback + 1 : i + 1]
        
        # Калибруем параметры OU на этом окне
        params = calibrate_ou(window)

        # Если спред сломался (ушел в жесткий тренд) — принудительно кроем позицию
        if params is None or not params['is_stationary']:
            state = 0.0
            hold = 0
            out[i] = state
            continue

        mu = params['mu']
        sigma = params['sigma_eq']
        current_px = sv[i]

        if np.isnan(sigma) or sigma == 0:
            out[i] = state
            continue

        # Истинный Z-score относительно математического среднего спреда
        zi = (current_px - mu) / sigma

        if state == 0.0:
            # Вход в позицию
            if zi >= entry:
                state = -min(abs(zi), stop)      # Спред слишком широкий -> шорт
                hold = 1
            elif zi <= -entry:
                state = min(abs(zi), stop)       # Спред слишком узкий -> лонг
                hold = 1
        else:
            hold += 1
            # Условия выхода
            reverted = abs(zi) <= exit
            blown = abs(zi) >= stop
            timed = (max_hold is not None) and (hold >= max_hold)
            
            if reverted or blown or timed:
                state = 0.0
                hold = 0
                
        out[i] = state

    pos[:] = out
    return pos


def ou_zscore(close, sma_window=20, z_window=20, entry=2.0, exit=0.5, stop=4.0, max_hold=None):
    """
    Оригинальная функция на основе скользящих средних.
    Оставлена для обратной совместимости с тестами на акциях.
    """
    mean = close.rolling(sma_window).mean()
    std = close.rolling(z_window).std()
    z = (close - mean) / std

    pos = pd.Series(0.0, index=close.index)
    state = 0.0          
    hold = 0             

    zv = z.values
    out = np.zeros(len(zv))

    for i in range(len(zv)):
        zi = zv[i]
        if np.isnan(zi):
            out[i] = 0.0
            state = 0.0
            hold = 0
            continue

        if state == 0.0:
            if zi >= entry:
                state = -min(abs(zi), stop)      
                hold = 1
            elif zi <= -entry:
                state = min(abs(zi), stop)       
                hold = 1
        else:
            hold += 1
            reverted = abs(zi) <= exit
            blown = abs(zi) >= stop
            timed = (max_hold is not None) and (hold >= max_hold)
            if reverted or blown or timed:
                state = 0.0
                hold = 0
        out[i] = state

    pos[:] = out
    return pos