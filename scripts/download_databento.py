import databento as db
import pandas as pd

client = db.Historical("db-m97t7U9Fe3PwS3vtrTnis9WmEYH9E")

data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=[
        "CL.c.0", "CL.c.1",
        "NG.c.0", "NG.c.1",
        "GC.c.0", "GC.c.1",
        "SI.c.0", "SI.c.1",
        "HG.c.0", "HG.c.1",
        "ZW.c.0", "ZW.c.1",
        "ZC.c.0", "ZC.c.1",
    ],
    stype_in="continuous",
    schema="ohlcv-1d",
    start="2015-01-01",
    end="2025-01-01",
)

df = data.to_df()
df = df[["symbol", "open", "high", "low", "close", "volume"]]

# Конвертация цен
price_cols = ["open", "high", "low", "close"]
df[price_cols] = df[price_cols] / 1_000_000_000

df.to_csv("/Users/shalygin/dev/Python_work/EMA1221/data/futures_m1_m2.csv")

# Проверка
for sym in ["CL.c.0", "GC.c.0", "NG.c.0", "ZW.c.0"]:
    row = df[df["symbol"] == sym].iloc[1]
    print(f"{sym}: close = {row['close']:.4f}")