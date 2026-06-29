import pandas as pd
import numpy as np

def donchian_seasonal(close, entry=20, exit_period=10):
    """
    Сезонный Дончиан (Август-Ноябрь).
    Адаптирован под экосистему: обработка NaN, оптимизация цикла, float-позиции.
    """
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()
    
    position = pd.Series(0.0, index=close.index)
    in_position = False
    
    # Предварительное извлечение месяцев для ускорения (вместо .month в цикле)
    months = close.index.month
    start_idx = max(entry, exit_period)
    
    for i in range(start_idx, len(close)):
        price = close.iloc[i]
        
        if in_position:
            if price < low_m.iloc[i]:
                in_position = False
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = 1.0
        else:
            # Сезонный фильтр: переход от инъекций к отбору
            if months[i] in (8, 9, 10, 11) and price > high_n.iloc[i]:
                in_position = True
                position.iloc[i] = 1.0
                
    return position


def donchian_seasonal_voltarget(close, entry=20, exit_period=10, target_vol=0.15, vol_window=60):
    """
    Сезонный Дончиан + Volatility Targeting.
    Учитывает механику run_engine: ребалансировка только при значимом 
    изменении веса (>10%), чтобы не "съесть" прибыль комиссиями (trade * cost).
    """
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()
    
    # Годовая волатильность (close-to-close)
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(vol_window).std() * np.sqrt(252)
    
    # Вес позиции (ограничиваем максимальное плечо до 2.0x для безопасности)
    weight = np.clip(target_vol / vol, 0, 2.0)
    weight = weight.fillna(1.0) # Fallback для первых дней
    
    position = pd.Series(0.0, index=close.index)
    in_position = False
    months = close.index.month
    start_idx = max(entry, exit_period, vol_window)
    
    for i in range(start_idx, len(close)):
        price = close.iloc[i]
        
        if in_position:
            if price < low_m.iloc[i]:
                in_position = False
                position.iloc[i] = 0.0
            else:
                prev_pos = position.iloc[i-1]
                # Rebalance only if weight changes by > 10% to avoid turnover bleed
                if prev_pos > 0 and abs(weight.iloc[i] - prev_pos) / prev_pos > 0.10:
                    position.iloc[i] = weight.iloc[i]
                else:
                    position.iloc[i] = prev_pos
        else:
            if months[i] in (8, 9, 10, 11) and price > high_n.iloc[i]:
                in_position = True
                position.iloc[i] = weight.iloc[i]
                
    return position


def donchian_seasonal_atr(close, entry=20, exit_period=10, atr_period=14, atr_mult=2.0):

    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit_period).min()
    
    tr = abs(close - close.shift(1))
    atr = tr.rolling(window=atr_period).mean()
    
    position = pd.Series(0.0, index=close.index)
    in_position = False
    stop_loss = 0.0
    months = close.index.month
    start_idx = max(entry, exit_period, atr_period)
    
    for i in range(start_idx, len(close)):
        price = close.iloc[i]
        
        if in_position:
            current_stop = price - (atr_mult * atr.iloc[i])
            if current_stop > stop_loss:
                stop_loss = current_stop
                
            if price < stop_loss or price < low_m.iloc[i]:
                in_position = False
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = 1.0
        else:
            if months[i] in (8, 9, 10, 11) and price > high_n.iloc[i]:
                in_position = True
                stop_loss = price - (atr_mult * atr.iloc[i])
                position.iloc[i] = 1.0
                
    return position

