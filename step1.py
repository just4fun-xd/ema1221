import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = "BTC-USD"
name = "Bitcoin"

data = yf.download(ticker, period="1y", interval="1d")
data.columns = data.columns.droplevel(1)
close = data["Close"]

ema12 = close.ewm(span=12, adjust=False).mean()
ema21 = close.ewm(span=21, adjust=False).mean()

result = pd.DataFrame({
    "Close": close,
    "EMA12": ema12,
    "EMA21": ema21
})

result["bull"] = result["EMA12"] > result["EMA21"]

result["signal"] = ""
bull = result["bull"]
prev = result["bull"].shift(1)

cross_up = bull & (prev == False)
cross_down = (bull == False) & (prev == True)

result.loc[cross_up, "signal"] = "BUY"
result.loc[cross_down, "signal"] = "SELL"

result["returns"] = result["Close"] / result["Close"].shift(1) - 1
result["strategy"] = result["returns"] * result["bull"].shift(1)
result["equity"] = (1 + result["strategy"]).cumprod()

result["peak"] = result["equity"].cummax()
result["drawdown"] = result["equity"] / result["peak"] - 1

max_dd = result["drawdown"].min()

print(result[["Close", "bull", "returns", "strategy", "equity"]].tail(15))
print(f"Total return: {(result['equity'].iloc[-1] - 1) * 100:.1f}%")
print(f"Max drawdown: {max_dd * 100:.1f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(result.index, result["Close"], label=name, color="black", linewidth=1)
ax1.plot(result.index, result["EMA12"], label="EMA12", color="blue")
ax1.plot(result.index, result["EMA21"], label="EMA21", color="red")
ax1.set_title(f"{name}: price and EMA 12/21")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(result.index, result["equity"], label="Equity", color="green")
ax2.set_title("Strategy equity curve")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
