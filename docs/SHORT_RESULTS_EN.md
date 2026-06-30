# SHORT_RESULTS_EN.md — Cross-Sectional Short Leg Research

Redesign of the short leg after the mirror-image L/S failure
(`ema_ensemble_long_short`, `donchian_breakout_ls`). Hypothesis from
`SHORT_REDESIGN.md`: a symmetric directional short against a structurally
rising asset is doomed; the correct form of a short is **cross-sectional
market-neutral** (rank the basket, long the top against short the bottom),
where market beta cancels between the two legs.

Basket: 19 mega-cap US equities, then widened to the full S&P 500 (499
usable names). Daily yfinance bars (clean, no roll problem). Period
2018-07 → 2025. Engine: new `run_portfolio` (portfolio-level weight
matrix; `run_engine` untouched). Sanity check: `run_portfolio`
reproduces `run_engine` on a single instrument (+151.5% / −21.7%, match
to 0.0%).

---

## ⭐ HEADLINE (TL;DR)

**The short-leg track is CLOSED. Cross-sectional momentum has no short
alpha on equities.** Decisive test (section 3.5): on the wide S&P 500
basket the market-neutral version returns −9.4% at beta −0.06. The
positive return of the net-long variant (tilt, +67%) turned out to be
market beta (+0.60), not a short signal. Zero out the beta and the
signal loses money.

The path to this conclusion ran through four layers, each with a mechanism:

1. **Mirror-image L/S** — structural failure (shorting a rising asset on
   the dip). −35% portfolio, 1/19 profitable.
2. **Cross-sectional, narrow basket (19 names)** — formally passes the
   tests, but the result is a concentrated bet on Nvidia/Tesla/AMD
   (top-3 = 66% of P&L), not an edge.
3. **Cross-sectional, wide basket, tilt** — concentration dissolved
   (top-3 = 6%), but +67% is beta, not alpha.
4. **Cross-sectional, wide basket, market-neutral** — −9.4% at beta ≈ 0.
   No alpha. Track closed.

Intermediate results on the narrow basket (for the record; final verdict
in section 3.5):

| Variant (19 names) | TotRet | Max DD | Beta | Spread 2022 | Tests |
|---|---:|---:|---:|---:|:---:|
| Base (symmetric MN) | +40.0% | −18.3% | +0.17 | +14.5% | 3/3 |
| 1: tilt + dyn-gross | +111.8% | −15.0% | +0.84* | +2.7% | 3/3 |
| 2: regime short | +92.6% | −19.7% | +0.65* | −1.8% | 2/3 |
| 3: vol-scaled MN | +26.7% | −19.0% | +0.14 | +12.7% | 3/3 |
| long-only (control) | +164.0% | −10.7% | — | — | — |

\* net-long variants: beta ≈ 0 not expected by construction.

**Why this is a valuable result, not a failure:** the project built an
evidence-based map of WHERE short alpha is absent and WHY (structural
equity uptrend bias + a momentum edge with no working short side). The
method's only useful property is tilt's low drawdown (−14.2% on the wide
basket), but that is risk management available more cheaply via long +
vol target. Confirms Alexander's thesis: "trend following can work, just
not in the case you programmed it for".

---

## 📘 PLAIN-LANGUAGE EXPLANATION

**Beta and alpha.** The market is an escalator going up. Beta is how much
that escalator carries you (beta 1 = you ride with it, beta 0 = you stand
beside it). Beta is not your skill — it's the market's work. Alpha is what
you added on top of the escalator through skill. If you stand beside it
(beta 0) and still rise, that rise is your alpha.

**What dual momentum does.** Each month we look at the whole basket and
rank stocks by how much they rose over the past six months. The strongest
we buy (long). The weakest we sell short (bet on their fall). The idea:
strong keeps rising, weak keeps falling, and the gap between them is our
profit — regardless of where the whole market goes.

**What we found.** The idea is elegant but doesn't work on equities.
Strong and weak stocks ultimately move too similarly — the gap between
them gives no stable income. Shorting weak stocks in a rising market is
almost always a loss, because even weak names rise over time with everyone
else.

## 💵 WHAT $100 BECOMES (≈7 years, S&P 500)

| $100 invested in | Becomes | Result |
|---|---:|---|
| market-neutral short (pure signal, beta 0) | **$90.60** | lost $9.40 — no alpha |
| tilt (net-long) | **$167.30** | +$67, but that's market (beta 0.60), not skill |
| just hold S&P 500 (benchmark) | ~$220+ | market beat every short version |

IMPORTANT: figures are inflated by survivorship bias (basket of surviving
companies) — real results are even more modest. Conclusion: the short leg
did not earn; its only benefit was tilt's low drawdown (−14%), paid for
with return.

---

## 1. What was tested, with mechanisms

### 1.1. Mirror-image L/S (closed earlier) — why it failed
`ema_ensemble_long_short`, `donchian_breakout_ls`: short = inverted long
signal. Shorts a structurally rising asset on the pullback → catches the
bounce up. −35% portfolio, 1/19 profitable. Not a form of short, but a
form of failure.

### 1.2. Cross-sectional market-neutral (base) — works, but expensive
Rank 19 names by 126-day momentum (dropping the last month against
short-term reversal), long top 20% against short bottom 20%, equal weight.
Absolute filter: do not short a name above its SMA200.
- Beta +0.17 — neutrality achieved (legs cancel the market).
- 2022 spread +14.5% — in the bear year turned long-only −5% into MN +9%.
- DD −18.3% — passes < 40% comfortably.
- BUT: total 40% vs 164% long-only. Spread negative every year except
  2022 → the leg is insurance you pay for every bull year.

### 1.3. "Hedge, not alpha" archetype (spread by year, narrow basket)

| Year | Base | 1: tilt | 2: regime | 3: vol-sc |
|---|---:|---:|---:|---:|
| 2018 | −1.4% | −0.1% | −0.1% | −3.1% |
| 2019 | −16.8% | −5.6% | −8.7% | −17.0% |
| 2020 | −15.9% | −1.0% | −3.4% | −20.4% |
| 2021 | −9.5% | −2.0% | 0.0% | −9.8% |
| **2022** | **+14.5%** | **+2.7%** | **−1.8%** | **+12.7%** |
| 2023 | −19.4% | −9.9% | −15.9% | −21.4% |
| 2024 | −13.1% | −4.0% | −1.0% | −14.1% |
| 2025 | −8.2% | −4.8% | −3.4% | −7.9% |

Spread = variant return minus long-only. One green column (2022) in seven
years — that is "hedge, not alpha" in numbers.

---

## 2. Three improvements — outcomes

### 2.1. Variant 1: tilt + dynamic gross — best risk-adjusted (narrow)
Asymmetric legs (long 1.0, short 0.5) + halve the short when the market is
above SMA200. Less paid for the hedge in bull years.
- Total +111.8%, 2022 spread still +2.7% (hedge preserved).
- Beta +0.84 — effectively a hedged long-only, not an MN strategy.

### 2.2. Variant 2: regime short — REJECTED (lag)
Short switches on only in risk-off (SPY < SMA200), else long-only.
- Fails test 2: 2022 spread −1.8%. By the time SPY broke SMA200 down, the
  best of the bear move had passed.
- Mechanism: regime-filter lag — same defect `EMA_cheat_sheet.md` recorded
  for the EMA200 filter, recurring at portfolio level.

### 2.3. Variant 3: vol-scaled MN — REDUNDANT
Inverse-volatility weighting within each leg. Profile ≈ base. On this
basket large names did not dominate leg risk anyway. Closed as redundant —
same story as %b and MACD earlier (a filter adding no independent signal).

---

## 3. Universal principles (confirmed)

- **Form of the short decides everything.** Mirror directional short vs an
  uptrend asset is structurally unprofitable. Cross-sectional market-neutral
  is the only form that passed the tests.
- **Neutrality ≠ profitability.** Beta ≈ 0 is easy; strong-minus-weak alpha
  on this basket appears only in a crisis.
- **The hedge has a large price.** Every bull year eats 8–20 pp vs
  long-only on a strongly trending market.
- **Regime filter lags** (repeat of the EMA200 finding).
- **Inverse-vol adds no signal** (repeat of the MACD / %b finding).

---

## 3.5. Wide-basket test (S&P 500) — DECISIVE

On the narrow basket (19 names), concentration appeared: 3 names
(Nvidia/Tesla/AMD) carried 66% of P&L; dropping two halved the return
112%→54%. Hypothesis: concentration is a narrow-basket artifact, not a
property of the method. Test: run on the full S&P 500 (499 usable names,
survivorship-biased — current members only, absolute numbers overstated).

**Result 1 — concentration dissolved (as expected):**
top-3 P&L share 66% → ~6%, ~174 of 499 names active. The wide universe
gives the method a real cross-section.

**Result 2 — but no alpha (the decisive test):**

| Variant (S&P 500) | TotRet | Max DD | Beta | top-3 P&L |
|---|---:|---:|---:|:---:|
| base (market-neutral) | **−9.4%** | −25.9% | −0.06 | 5.3% |
| tilt (net-long) | +67.3% | −14.2% | +0.60 | 5.9% |

Market-neutral at beta −0.06 returns **−9.4%** — not zero, negative. So
tilt's 67% was **market beta** (+0.60 × a rising market), NOT short alpha.
Zero the beta and the long-minus-short signal loses money. With
survivorship bias removed, worse still.

**Section conclusion:** cross-sectional momentum on daily S&P 500 equities
over 2018–2025 has NO short alpha. The method delivers only (a) market
exposure, cheaper via a plain long, and (b) tilt's low drawdown (−14.2%) —
risk management, not return. This definitively closes the equity short-leg
track.

---

## 4. Verdict and recommendation

**The equity short-leg track is CLOSED.** Deliver as completed research
with an evidence-based map of limitations — not a production strategy.

What the full cycle showed:
- No short alpha on daily S&P 500 equities 2018–2025 (market-neutral
  −9.4% at beta ≈ 0, section 3.5).
- The only useful property is net-long tilt's low drawdown (−14.2%), but
  that is risk management available more cheaply via long + vol target.
- Intermediate variants (base MN, regime, vol-scaled) closed with
  mechanisms: regime — filter lag; vol-scaled — redundant; base — neutral
  but alpha rode on narrow-basket concentration that vanishes when wide.

**Main honest takeaway:** a short leg on a structurally rising equity
basket creates no return — on a narrow basket it masquerades as
concentration in leaders, on a wide basket it is exposed as negative
alpha. This is a map of the method's limits, alongside the project's
findings on vol targeting, pairs trading and carry.

### Next logical step (not now)
Move to commodities, where there is no structural uptrend bias and
cross-sectional momentum dispersion is higher — there a short leg could
in theory have the alpha it lacks on equities. BUT: requires
roll-adjusted continuous futures (Databento `futures_m1_m2_fixed.csv`),
else roll gaps create false momentum. A separate hypothesis, not a
continuation of this one.

---

## 5. Files

- `strategies/strategies_dualmom.py` — `dual_momentum` (base),
  `_long_only` (control), `_tilt`, `_regime`, `_volscaled`.
- `core/engine_portfolio.py` — `run_portfolio`, `load_basket`,
  `positions_to_weights` (adapter for per-series strategies),
  `portfolio_beta`, `run_portfolio_yearly` (per-instrument yearly report).
- `core/universe.py` — S&P 500 loader (`get_sp500_tickers`,
  `load_basket_batch`).
- `runners/run_dualmom.py` — variants + three honesty tests + yearly report.
- `runners/run_wide.py` — wide-basket market-neutral vs tilt comparison.
- `runners/compare_all.py` — cross-sectional vs per-series on one basket.
- `runners/tilt_yearly.py` — per-instrument yearly report for tilt.
- `diagnostics/_concentration_check.py` — P&L concentration by name.

Reproduce: `python -m runners.run_wide`. The `run_engine` and all
per-series strategies are untouched — numbers in `BENCHMARK_RESULTS.md`
remain valid.
