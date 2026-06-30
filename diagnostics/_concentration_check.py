"""Diagnostic: how concentrated is dual_momentum_tilt's P&L?

Answers two separate questions behind "it barely trades":
  1. How many names are active at once? (quintiles => ~8 of 19 is normal)
  2. Does the RESULT lean on 1-2 names (Tesla/Nvidia)? (the real risk)

Run locally: python _concentration_check.py
"""
import pandas as pd
import numpy as np
from core.engine import load_data
from core.engine_portfolio import load_basket, run_portfolio
from strategies.strategies_dualmom import dual_momentum_tilt

BASKET = {"S&P 500":"SPY","Nasdaq":"QQQ","Apple":"AAPL","Tesla":"TSLA",
  "Nvidia":"NVDA","Microsoft":"MSFT","Amazon":"AMZN","Meta":"META",
  "Alphabet":"GOOGL","AMD":"AMD","Netflix":"NFLX","Visa":"V",
  "Mastercard":"MA","UnitedHealth":"UNH","Home Depot":"HD",
  "Bank of America":"BAC","Coca-Cola":"KO","Procter & Gamble":"PG",
  "Walmart":"WMT"}

LOAD_START, END, TRADE_START, COST = "2017-01-01", "2026-01-01", "2018-07-01", 0.001

prices = load_basket(BASKET, LOAD_START, END)
spy = load_data("SPY", LOAD_START, END)
w = dual_momentum_tilt(prices, benchmark=spy)
held = w.shift(1)

# --- Q1: how many names active at once ---
active = (held.abs() > 1e-9).sum(axis=1)
mask = held.index >= TRADE_START
print("=" * 56)
print("Q1: ACTIVITY (how many names hold a position)")
print("=" * 56)
print(f"  active names per day: avg {active[mask].mean():.1f}, "
      f"min {int(active[mask].min())}, max {int(active[mask].max())}")
print(f"  ({prices.shape[1]} instruments total; "
      f"quintile top+bottom ~8 is by design)\n")

# --- Q2: P&L contribution per name ---
asset_ret = prices.pct_change()
contrib = (asset_ret * held)[mask]           # daily pnl per name
total_per_name = contrib.sum().sort_values(ascending=False)
gross = total_per_name.abs().sum()

print("=" * 56)
print("Q2: P&L CONCENTRATION (does result lean on a few names)")
print("=" * 56)
print(f"{'Instrument':<18}{'PnL share':>12}{'days held':>12}")
print("-" * 56)
day_frac = (held.abs() > 1e-9)[mask].mean()
for n, v in total_per_name.items():
    print(f"  {n:<16}{v/gross*100:>10.1f}%{day_frac[n]*100:>11.1f}%")

top1 = total_per_name.abs().sort_values(ascending=False)
print("-" * 56)
print(f"  top-1 name = {top1.index[0]}: "
      f"{top1.iloc[0]/gross*100:.0f}% of gross P&L")
print(f"  top-3 names = {top1.iloc[:3].sum()/gross*100:.0f}% of gross P&L")

# --- Q3: drop the top-2 winners, re-run ---
winners = top1.index[:2].tolist()
keep = [c for c in prices.columns if c not in winners]
w2 = dual_momentum_tilt(prices[keep], benchmark=spy)
_, r_full, dd_full = run_portfolio(prices, w, TRADE_START, COST, target_vol=0.10)
_, r_drop, dd_drop = run_portfolio(prices[keep], w2, TRADE_START, COST, target_vol=0.10)
print("\n" + "=" * 56)
print("Q3: ROBUSTNESS (drop the 2 biggest names, re-run)")
print("=" * 56)
print(f"  full basket:        ret {r_full*100:+7.1f}%  dd {dd_full*100:6.1f}%")
print(f"  without {winners[0]}/{winners[1]}: ret {r_drop*100:+7.1f}%  dd {dd_drop*100:6.1f}%")
drop = (r_full - r_drop) * 100
print(f"  => removing 2 names changes total return by {drop:+.0f} pp")
print("  (if this wipes most of the return, the edge is those 2 names)")