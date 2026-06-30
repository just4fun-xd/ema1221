import pandas as pd

df = pd.read_csv(
    "/Users/shalygin/dev/Python_work/EMA1221/data/futures_m1_m2_fixed.csv",
    index_col=0,
    parse_dates=True
)

# Разделим на M1 и M2
m1 = df[df["symbol"].str.endswith(".c.0")].pivot(columns="symbol", values="close")
m2 = df[df["symbol"].str.endswith(".c.1")].pivot(columns="symbol", values="close")

# Спред (term structure)
spread = m1.values - m2.values
spread_df = pd.DataFrame(spread, index=m1.index, columns=[s.replace(".c.0", "") for s in m1.columns])

print(spread_df.head(10))
print(f"\nФорма: {spread_df.shape}")
