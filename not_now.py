import pandas as pd


def donchian_ensemble_macd_4step_pyramid(
    close, pairs=None, threshold=0.5, entry=20, exit=10, 
    target_vol=0.15, vol_window=30, max_pos=2.0
):
    """
    Шаг 1. Истинная (убывающая) пирамида: 40% - 30% - 20% - 10%.
    Снижает среднюю цену позиции, защищая прибыль при разворотах.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]
        
    # --- Блок 1: Расчет индикаторов ---
    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)
        
    trend_score = votes.mean(axis=1)
    macro_bullish = trend_score > threshold

    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    macd_bullish = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))

    atr = (close - close.shift(1)).abs().rolling(14).mean()
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit).min()

    # --- Блок 2: Стейт-машина (Убывающий объем) ---
    raw_position = pd.Series(0.0, index=close.index)
    
    # Массив накопленного объема: 0, 40%, 70%, 90%, 100%
    cumulative_sizes = [0.0, 0.4, 0.7, 0.9, 1.0]
    
    in_position = False
    position_units = 0
    last_entry_price = 0.0

    for i in range(len(close)):
        price = close.iloc[i]
        cur_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.0

        if in_position:
            # Динамический стоп подтягивается за последней покупкой
            turtle_stop = last_entry_price - 2.0 * cur_atr
            donchian_stop = low_m.iloc[i]
            final_stop = max(turtle_stop, donchian_stop)

            if price < final_stop:
                in_position = False
                position_units = 0
                last_entry_price = 0.0
            else:
                # Доливка: шаг 0.8 ATR
                scale_condition = price >= (last_entry_price + 0.8 * cur_atr)
                if position_units < 4 and scale_condition:
                    position_units += 1
                    last_entry_price = price
                
                # Берем объем из нашего нового убывающего массива
                raw_position.iloc[i] = cumulative_sizes[position_units]
        else:
            if (price > high_n.iloc[i] and 
                    macro_bullish.iloc[i] and macd_bullish.iloc[i]):
                in_position = True
                position_units = 1
                last_entry_price = price
                raw_position.iloc[i] = cumulative_sizes[position_units] # 0.4

    # --- Блок 3: VolTarget ---
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (256 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)

    return raw_position * size


def donchian_ensemble_macd_4step_time(
    close, pairs=None, threshold=0.5, entry=20, exit=10, 
    target_vol=0.15, vol_window=30, max_pos=2.0, scale_delay=5
):
    """
    Шаг 2. Убывающая пирамида (40-30-20-10) + Временной фильтр доливок.
    Запрещает покупать следующую ступень раньше, чем через `scale_delay` дней.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]
        
    # --- Блок 1: Расчет индикаторов ---
    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)
        
    trend_score = votes.mean(axis=1)
    macro_bullish = trend_score > threshold

    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    macd_bullish = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))

    atr = (close - close.shift(1)).abs().rolling(14).mean()
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit).min()

    # --- Блок 2: Стейт-машина (Убывающий объем + Time Spacing) ---
    raw_position = pd.Series(0.0, index=close.index)
    
    cumulative_sizes = [0.0, 0.4, 0.7, 0.9, 1.0]
    
    in_position = False
    position_units = 0
    last_entry_price = 0.0
    last_entry_index = 0  # Запоминаем номер дня последней сделки

    for i in range(len(close)):
        price = close.iloc[i]
        cur_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.0

        if in_position:
            # Динамический стоп подтягивается за последней покупкой
            turtle_stop = last_entry_price - 2.0 * cur_atr
            donchian_stop = low_m.iloc[i]
            final_stop = max(turtle_stop, donchian_stop)

            if price < final_stop:
                in_position = False
                position_units = 0
                last_entry_price = 0.0
                last_entry_index = 0
            else:
                # Проверяем, прошло ли заданное количество дней
                time_condition = (i - last_entry_index) >= scale_delay
                scale_condition = price >= (last_entry_price + 0.8 * cur_atr)
                
                if position_units < 4 and scale_condition and time_condition:
                    position_units += 1
                    last_entry_price = price
                    last_entry_index = i  # Обновляем день последней покупки
                
                raw_position.iloc[i] = cumulative_sizes[position_units]
        else:
            if (price > high_n.iloc[i] and 
                    macro_bullish.iloc[i] and macd_bullish.iloc[i]):
                in_position = True
                position_units = 1
                last_entry_price = price
                last_entry_index = i  # Запоминаем день первого входа
                raw_position.iloc[i] = cumulative_sizes[position_units]

    # --- Блок 3: VolTarget ---
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (256 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)

    return raw_position * size


def donchian_ensemble_macd_4step_take(
    close, pairs=None, threshold=0.5, entry=20, exit=10,
    target_vol=0.15, vol_window=30, max_pos=2.0,
    scale_delay=5, take_atr_mult=3.5
):
    """
    Шаг 3. Убывающая пирамида (40-30-20-10) + Time Spacing + Тейк-Профит.
    Сбрасывает 4-ю ступень при достижении ценой +3.5 ATR от ее входа.
    """
    if pairs is None:
        pairs = [(5, 20), (10, 40), (20, 80), (40, 160), (64, 256)]

    # --- Блок 1: Расчет индикаторов ---
    votes = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        votes[f"{fast}/{slow}"] = (ema_fast > ema_slow).astype(int)

    trend_score = votes.mean(axis=1)
    macro_bullish = trend_score > threshold

    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    # Скобки обязательны для побитовых операторов в pandas
    macd_bullish = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))

    atr = (close - close.shift(1)).abs().rolling(14).mean()
    high_n = close.shift(1).rolling(entry).max()
    low_m = close.shift(1).rolling(exit).min()

    # --- Блок 2: Стейт-машина ---
    raw_position = pd.Series(0.0, index=close.index)
    cumulative_sizes = [0.0, 0.4, 0.7, 0.9, 1.0]

    in_position = False
    position_units = 0
    last_entry_price = 0.0
    last_entry_index = 0

    for i in range(len(close)):
        price = close.iloc[i]
        cur_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0.0

        if in_position:
            # 2.1. Расчет защитных стопов
            turtle_stop = last_entry_price - 2.0 * cur_atr
            donchian_stop = low_m.iloc[i]
            final_stop = max(turtle_stop, donchian_stop)

            if price < final_stop:
                # Полный выход по стопу
                in_position = False
                position_units = 0
                last_entry_price = 0.0
                last_entry_index = 0
            else:
                # 2.2. Логика Тейк-Профита (Scaling Out)
                if position_units == 4:
                    take_level = last_entry_price + take_atr_mult * cur_atr
                    if price >= take_level:
                        position_units = 3  # Сброс верхней ступени в кэш
                        # last_entry_price не меняем, чтобы жесткий стоп 
                        # оставался подтянутым к максимумам

                # 2.3. Логика дозакупки (Scale-In)
                time_ok = (i - last_entry_index) >= scale_delay
                scale_ok = price >= (last_entry_price + 0.8 * cur_atr)

                if position_units < 4 and scale_ok and time_ok:
                    position_units += 1
                    last_entry_price = price
                    last_entry_index = i

                raw_position.iloc[i] = cumulative_sizes[position_units]
        else:
            # 2.4. Поиск точки входа
            if (price > high_n.iloc[i] and
                    macro_bullish.iloc[i] and macd_bullish.iloc[i]):
                in_position = True
                position_units = 1
                last_entry_price = price
                last_entry_index = i
                raw_position.iloc[i] = cumulative_sizes[position_units]

    # --- Блок 3: Риск-менеджмент (VolTarget) ---
    daily_vol = close.pct_change().rolling(vol_window).std()
    annual_vol = daily_vol * (256 ** 0.5)
    size = (target_vol / annual_vol).clip(upper=max_pos)

    return raw_position * size