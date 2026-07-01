"""HMM regime diagnostic: are the regimes STABLE and TIMELY?

Trek 2.2, step 0. Before building a regime classifier or a router, answer
two questions that decide whether HMM is worth building at all:

  Q1 STABILITY -- do the hidden states mean the same thing out-of-sample?
     Fit HMM on TRAIN, map each state to an interpretable label by its
     emission stats (high vol -> crisis, positive drift + low vol ->
     trend, ~0 drift + low vol -> range). Refit on TEST. If the state
     with "crisis" characteristics on train also has them on test, the
     regimes are stable. If the labels scramble, HMM is overfitting the
     period -- close it cheap, like OU.

  Q2 TIMELINESS -- does the regime arrive on time, or lagged?
     A stable regime is useless to a router if it flags "trend" a month
     after the trend started (the EMA200-filter lag that failed the
     project repeatedly). We overlay the forward-filtered regime
     probability on realised forward vol and check that the crisis state
     rises AT the vol spike, not after it.

Look-ahead discipline: the router would use forward filtering
(P(state_t | data up to t)), NOT Viterbi over the whole series (which
peeks at the future). We report BOTH here only to show the gap -- the
router must use the forward version.

Features (kept minimal + interpretable, so states are checkable):
  - daily log return
  - rolling realised vol (20d)

Run:  python -m diagnostics.hmm_regime
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


N_STATES = 3
VOL_WINDOW = 20
SEED = 42
TRAIN = ("2020-01-01", "2023-01-01")
TEST = ("2023-01-01", "2025-01-01")


def make_features(px):
    """Return a (T, 2) feature frame: log return, rolling realised vol."""
    logret = np.log(px / px.shift(1))
    vol = logret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    feat = pd.DataFrame({"ret": logret, "vol": vol}).dropna()
    return feat


def fit_hmm(feat, n_states=N_STATES, seed=SEED):
    X = feat.values
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=200, random_state=seed)
    model.fit(X)
    return model


def label_states(model):
    """Map each state index -> interpretable label from its emission means.

    Returns dict {state_idx: label} using:
      highest vol  -> 'crisis'
      of the rest: highest |mean return| with positive drift -> 'trend'
                   the remaining low-vol state -> 'range'
    """
    means = model.means_          # shape (n_states, 2): [ret, vol]
    ret_m, vol_m = means[:, 0], means[:, 1]
    order_vol = np.argsort(vol_m)          # ascending vol
    crisis = order_vol[-1]                 # highest vol
    rest = [s for s in range(len(vol_m)) if s != crisis]
    # among the rest, the one with larger |drift| is 'trend'
    if abs(ret_m[rest[0]]) >= abs(ret_m[rest[1]]):
        trend, rng = rest[0], rest[1]
    else:
        trend, rng = rest[1], rest[0]
    return {crisis: "crisis", trend: "trend", rng: "range"}


def regime_signature(model, feat):
    """Human-readable per-state stats to eyeball stability across periods."""
    labels = label_states(model)
    states = model.predict(feat.values)
    rows = []
    for s in range(model.n_components):
        mask = states == s
        share = mask.mean()
        rows.append((labels[s],
                     model.means_[s, 0] * 252,      # annualised drift
                     model.means_[s, 1],            # annualised vol (already)
                     share))
    return sorted(rows, key=lambda r: r[0])   # by label


def print_signature(title, sig):
    print(f"\n  {title}")
    print(f"    {'label':<8}{'drift/yr':>10}{'vol/yr':>9}{'time%':>8}")
    for label, drift, vol, share in sig:
        print(f"    {label:<8}{drift*100:>9.1f}%{vol*100:>8.1f}%"
              f"{share*100:>7.1f}%")


def timeliness(px, model, feat):
    """Does the crisis regime rise AT the vol spike, or lag it?

    Forward-filtered P(crisis) vs realised vol. We report the correlation
    of P(crisis) with vol at lag 0 and the best lag; a router needs the
    signal concurrent or leading, not trailing.
    """
    labels = label_states(model)
    crisis_state = [s for s, l in labels.items() if l == "crisis"][0]
    # forward filtering: posteriors given data up to each t
    post = model.predict_proba(feat.values)
    p_crisis = pd.Series(post[:, crisis_state], index=feat.index)
    vol = feat["vol"]

    best_lag, best_corr = 0, -2
    for lag in range(-10, 11):          # +lag = p_crisis lags vol
        c = p_crisis.shift(lag).corr(vol)
        if c is not None and c > best_corr:
            best_corr, best_lag = c, lag
    corr0 = p_crisis.corr(vol)
    return corr0, best_lag, best_corr


def run_period(px, tag):
    feat = make_features(px)
    model = fit_hmm(feat)
    sig = regime_signature(model, feat)
    print_signature(tag, sig)
    return model, feat


def main(px=None):
    if px is None:
        # Real data path -- adjust to your loader (single liquid instrument
        # first, e.g. CL front month from panels).
        from core.load_panels import load_panels
        P = load_panels()
        px = P["close"]["CL"].where(P["native"]["CL"]).dropna()
        print(f"Instrument: CL, {len(px)} days")

    tr = px[(px.index >= TRAIN[0]) & (px.index < TRAIN[1])]
    te = px[(px.index >= TEST[0]) & (px.index < TEST[1])]

    print("=" * 60)
    print("  Q1  STABILITY: same states, two periods")
    print("=" * 60)
    m_tr, f_tr = run_period(tr, f"TRAIN {TRAIN[0]}..{TRAIN[1]}")
    m_te, f_te = run_period(te, f"TEST  {TEST[0]}..{TEST[1]}")

    print("\n  Read: if 'crisis' has high vol/yr and 'trend' has clear")
    print("  drift on BOTH periods, regimes are stable. If a label's")
    print("  drift/vol flip sign or magnitude across periods -> unstable.")

    print("\n" + "=" * 60)
    print("  Q2  TIMELINESS: does P(crisis) track vol on time?")
    print("=" * 60)
    for tag, m, f in [("TRAIN", m_tr, f_tr), ("TEST", m_te, f_te)]:
        corr0, best_lag, best_corr = timeliness(None, m, f)
        verdict = ("concurrent/leading" if best_lag <= 0
                   else f"LAGS by {best_lag}d")
        print(f"  {tag}: corr(P_crisis, vol) @0 = {corr0:+.2f}; "
              f"best lag {best_lag:+d}d (corr {best_corr:+.2f}) -> {verdict}")
    print("\n  best_lag <= 0 = regime is concurrent or leads vol (good for")
    print("  a router). best_lag > 0 = P(crisis) trails the vol spike")
    print("  (the EMA200-filter lag failure -> router would be late).")


if __name__ == "__main__":
    main()
