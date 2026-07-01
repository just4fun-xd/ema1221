"""HMM regime diagnostic v2 -- fixed features.

v1 was TIMELY (P_crisis tracked vol at lag 0) but UNSTABLE across periods:
  - 'crisis' meant vol 120%/8%-of-time on train but 32%/54% on test,
    because HMM learned ABSOLUTE vol, and the vol level drifts year to
    year (2020 shock vs calmer 2023).
  - 'trend' and 'range' had identical vol (~21%) on test, so HMM couldn't
    separate them -- fatal for a router that must pick Donchian vs OU.

Two feature fixes:
  1. NORMALISED vol = vol / SMA(vol). "High vol" now means "high vs its
     own recent baseline", invariant to the drifting absolute level.
     Fixes the crisis-label instability.
  2. EFFICIENCY RATIO = |sum(returns)| / sum(|returns|) over a window.
     ~1 = price moved efficiently one way (trend); ~0 = lots of motion,
     little net progress (range). Independent of vol -> separates
     trend from range without relying on drift.

Everything else (fit on train & test separately, label by emissions,
timeliness via forward filtering) is unchanged from v1, so the
before/after comparison is clean.

Run:  python -m diagnostics.hmm_regime_v2
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


N_STATES = 3
VOL_WINDOW = 20
VOL_BASELINE = 60        # SMA window for normalising vol
ER_WINDOW = 20           # efficiency-ratio window
SEED = 42
TRAIN = ("2020-01-01", "2023-01-01")
TEST = ("2023-01-01", "2025-01-01")


def make_features(px):
    """(T, 3) features: log return, normalised vol, efficiency ratio.

    - ret: daily log return (drift/direction).
    - nvol: realised vol / its own SMA -> relative, not absolute.
    - er: |net move| / |gross move| over ER_WINDOW -> trendiness.
    """
    logret = np.log(px / px.shift(1))
    vol = logret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    nvol = vol / vol.rolling(VOL_BASELINE).mean()

    net = px.diff(ER_WINDOW).abs()
    gross = px.diff().abs().rolling(ER_WINDOW).sum()
    er = (net / gross).replace([np.inf, -np.inf], np.nan)

    feat = pd.DataFrame({"ret": logret, "nvol": nvol, "er": er}).dropna()
    return feat


def fit_hmm(feat, n_states=N_STATES, seed=SEED):
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=200, random_state=seed)
    model.fit(feat.values)
    return model


def label_states(model):
    """Map state -> label using emission means [ret, nvol, er].

    crisis = highest normalised vol.
    of the other two: higher efficiency ratio = trend, lower = range.
    (er, not drift, now decides trend vs range -- the v1 fix.)
    """
    means = model.means_                 # (n_states, 3): ret, nvol, er
    nvol_m, er_m = means[:, 1], means[:, 2]
    crisis = int(np.argmax(nvol_m))
    rest = [s for s in range(len(nvol_m)) if s != crisis]
    if er_m[rest[0]] >= er_m[rest[1]]:
        trend, rng = rest[0], rest[1]
    else:
        trend, rng = rest[1], rest[0]
    return {crisis: "crisis", trend: "trend", rng: "range"}


def regime_signature(model, feat):
    labels = label_states(model)
    states = model.predict(feat.values)
    rows = []
    for s in range(model.n_components):
        mask = states == s
        rows.append((labels[s],
                     model.means_[s, 0] * 252,     # annualised drift
                     model.means_[s, 1],           # normalised vol (~1 = avg)
                     model.means_[s, 2],           # efficiency ratio
                     mask.mean()))
    return sorted(rows, key=lambda r: r[0])


def print_signature(title, sig):
    print(f"\n  {title}")
    print(f"    {'label':<8}{'drift/yr':>10}{'nvol':>7}{'eff':>7}{'time%':>8}")
    for label, drift, nvol, er, share in sig:
        print(f"    {label:<8}{drift*100:>9.1f}%{nvol:>7.2f}{er:>7.2f}"
              f"{share*100:>7.1f}%")


def timeliness(model, feat):
    labels = label_states(model)
    crisis_state = [s for s, l in labels.items() if l == "crisis"][0]
    post = model.predict_proba(feat.values)
    p_crisis = pd.Series(post[:, crisis_state], index=feat.index)
    vol = feat["nvol"]
    best_lag, best_corr = 0, -2
    for lag in range(-10, 11):
        c = p_crisis.shift(lag).corr(vol)
        if c is not None and c > best_corr:
            best_corr, best_lag = c, lag
    return p_crisis.corr(vol), best_lag, best_corr


def run_period(px, tag):
    feat = make_features(px)
    model = fit_hmm(feat)
    print_signature(tag, regime_signature(model, feat))
    return model, feat


def main(px=None):
    if px is None:
        from core.load_panels import load_panels
        P = load_panels()
        px = P["close"]["CL"].where(P["native"]["CL"]).dropna()
        print(f"Instrument: CL, {len(px)} days")

    tr = px[(px.index >= TRAIN[0]) & (px.index < TRAIN[1])]
    te = px[(px.index >= TEST[0]) & (px.index < TEST[1])]

    print("=" * 62)
    print("  Q1  STABILITY (v2 features): same states, two periods")
    print("=" * 62)
    m_tr, f_tr = run_period(tr, f"TRAIN {TRAIN[0]}..{TRAIN[1]}")
    m_te, f_te = run_period(te, f"TEST  {TEST[0]}..{TEST[1]}")
    print("\n  nvol ~1 = average vol for the period; >1.5 = elevated.")
    print("  eff (0..1): high = trending, low = choppy/range.")
    print("  STABLE if: crisis has highest nvol on BOTH periods AND")
    print("  trend has higher eff than range on BOTH periods.")

    print("\n" + "=" * 62)
    print("  Q2  TIMELINESS: P(crisis) vs normalised vol")
    print("=" * 62)
    for tag, m, f in [("TRAIN", m_tr, f_tr), ("TEST", m_te, f_te)]:
        c0, lag, cbest = timeliness(m, f)
        verdict = "concurrent/leading" if lag <= 0 else f"LAGS {lag}d"
        print(f"  {tag}: corr@0 {c0:+.2f}; best lag {lag:+d}d "
              f"(corr {cbest:+.2f}) -> {verdict}")


if __name__ == "__main__":
    main()
