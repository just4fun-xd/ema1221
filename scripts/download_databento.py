"""
Databento download: wide commodity basket for cross-sectional momentum.
22 liquid GLBX.MDP3 roots across 4 sectors, front (c.0) + second (c.1) month.
c.1 is pulled to compute roll yield / carry component separately from price trend.

NOTE: GLBX.MDP3 covers CME/NYMEX/COMEX/CBOT only. ICE softs (Sugar SB,
Coffee KC, Cocoa CC, Cotton CT, OJ) are NOT available here — need IFUS.
"""

import databento as db
import pandas as pd
import time


# Key from env — do not hardcode. export DATABENTO_API_KEY=...
client = db.Historical("db-jB4qujXTReCcYXxtkKNPh3UrNFXdx")

# --- Roster: 22 roots, grouped by sector -------------------------------------
ROOTS = {
    # Energy (4) — RB = gasoline (roadmap OU candidate), HO = heating oil
    "energy":    ["CL", "NG", "HO", "RB"],
    # Metals (5)
    "metals":    ["GC", "SI", "HG", "PL", "PA"],
    # Grains / oilseeds (7)
    "grains":    ["ZW", "KE", "ZC", "ZS", "ZM", "ZL", "ZO"],
    # Livestock (3)
    "livestock": ["LE", "GF", "HE"],
    # Rice (1) — thin, may drop after liquidity check
    "other":     ["ZR"],
}
# Total = 4+5+7+3+1 = 20 roots  (drop ZR if volume too low -> 19)

roots_flat = [r for sector in ROOTS.values() for r in sector]

# Roll method:
#   .c.n = calendar roll (front expiry). Fine for CL/NG/grains that trade
#          every month.
#   .v.n = volume roll (jumps to most-liquid contract). Cleaner for metals,
#          whose liquidity concentrates in specific months (GC: G,J,M,Q,Z),
#          so calendar front lands in thin months and prints price jumps.
VOLUME_ROLL = {"GC", "SI", "HG", "PL", "PA"}


def sym(root, n):
    stype = "v" if root in VOLUME_ROLL else "c"
    return f"{root}.{stype}.{n}"


# Build front + second month for each root
symbols = []
for r in roots_flat:
    symbols += [sym(r, 0), sym(r, 1)]

print(f"Requesting {len(roots_flat)} roots, {len(symbols)} symbols in batches")

# One big request times out the gateway (504). Fetch per-sector with retries,
# then concatenate. Each sector = 8-14 symbols, well within gateway limits.


def fetch_batch(batch_symbols, tries=4):
    for attempt in range(1, tries + 1):
        try:
            d = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=batch_symbols,
                stype_in="continuous",
                schema="ohlcv-1d",
                start="2020-01-01",
                end="2025-01-01",
            )
            return d.to_df()
        except db.common.error.BentoServerError as e:
            wait = 5 * attempt
            print(f"    {e.__class__.__name__} (attempt {attempt}/{tries}) "
                  f"-> retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"batch failed after {tries} tries: {batch_symbols}")


frames = []
for sector, roots in ROOTS.items():
    batch = []
    for r in roots:
        batch += [sym(r, 0), sym(r, 1)]
    print(f"  [{sector}] {len(batch)} symbols...")
    frames.append(fetch_batch(batch))

df = pd.concat(frames)
df = df[["symbol", "open", "high", "low", "close", "volume"]]
# NOTE: to_df() already returns real prices in this client version.
# Confirmed: CL 71.87, GC 2627.5, SI 28.985, HG 3.982. Do NOT divide.

out = "/Users/shalygin/dev/Python_work/EMA1221/data/futures_basket_wide.csv"
df.to_csv(out)
print(f"Saved -> {out}")

# --- Liquidity + coverage audit ----------------------------------------------
# Catch dead/thin symbols BEFORE they contaminate the cross-sectional ranking.
print("\nPer-symbol audit (front month only):")
print(f"{'symbol':<8}{'rows':>7}{'first':>13}{'last':>13}{'med_vol':>12}{'close':>12}")
for r in roots_flat:
    s0 = sym(r, 0)
    s = df[df["symbol"] == s0]
    if len(s) == 0:
        print(f"{s0:<8}{'MISSING — check root / not on GLBX':>50}")
        continue
    print(
        f"{s0:<8}{len(s):>7}"
        f"{str(s.index.min().date()):>13}{str(s.index.max().date()):>13}"
        f"{s['volume'].median():>12.0f}{s['close'].iloc[-1]:>12.4f}"
    )

# --- Carry / roll-yield component (M1 vs M2) ---------------------------------
# This is the piece we agreed to log SEPARATELY from price momentum.
# roll_yield > 0  -> backwardation (front richer)  -> structural tailwind long
# roll_yield < 0  -> contango      (front cheaper) -> structural headwind long
print("\nRoll-yield snapshot (last obs, (M1-M2)/M2):")
for r in roots_flat:
    m1 = df[df["symbol"] == sym(r, 0)]["close"]
    m2 = df[df["symbol"] == sym(r, 1)]["close"]
    if len(m1) == 0 or len(m2) == 0:
        continue
    j = m1.index.intersection(m2.index)
    if len(j) == 0:
        continue
    ry = (m1.loc[j].iloc[-1] - m2.loc[j].iloc[-1]) / m2.loc[j].iloc[-1]
    tag = "backwardation" if ry > 0 else "contango"
    print(f"{r:<5}{ry:>+8.4f}  {tag}")
