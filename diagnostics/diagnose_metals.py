"""
Diagnose two issues from the wide-basket pull:
  1. price scaling (drop /1e9)
  2. metals resolving to wrong/thin contracts (GC/SI/HG med_vol implausibly low)

Run this BEFORE re-downloading the full basket.
"""

import os
import databento as db

client = db.Historical("db-jB4qujXTReCcYXxtkKNPh3UrNFXdx")

# --- 1. What do the metals continuous symbols actually resolve to? -----------
# Map continuous front-month -> the raw instrument it points at over time.
for root in ["GC", "SI", "HG", "PA", "PL", "CL"]:  # CL as a known-good control
    try:
        res = client.symbology.resolve(
            dataset="GLBX.MDP3",
            symbols=[f"{root}.c.0"],
            stype_in="continuous",
            stype_out="raw_symbol",
            start_date="2024-06-01",
            end_date="2024-07-01",
        )
        print(f"{root}.c.0 ->")
        for k, v in res["result"].items():
            print("   ", v)
    except Exception as e:
        print(f"{root}.c.0 -> ERROR: {e}")

# --- 2. Confirm real price magnitude WITHOUT the /1e9 scaling -----------------
print("\nRaw close (no /1e9), one recent day per root:")
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["CL.c.0", "GC.c.0", "SI.c.0", "HG.c.0", "NG.c.0"],
    stype_in="continuous",
    schema="ohlcv-1d",
    start="2024-12-20",
    end="2025-01-01",
)
d = data.to_df()
for sym in d["symbol"].unique():
    last = d[d["symbol"] == sym]["close"].iloc[-1]
    print(f"  {sym:<8} close = {last}")
# Expected ballpark: CL ~70, GC ~2600, SI ~29, HG ~4, NG ~3.5
# If these match -> to_df() already returns real prices -> remove /1e9.