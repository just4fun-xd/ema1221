"""Crisis-filtered Donchian vs plain Donchian -- does the filter pay?

The P&L-by-regime decomposition showed, on CL: Donchian earns +6-7% in
both trend AND range, but LOSES -5.6% (sharpe -1.21) in the HMM 'crisis'
regime. So the fix is not a 3-way router (trend/range don't differ) --
it's ONE patch: hold Donchian normally, go to CASH when HMM says crisis.

This runner builds that and compares:
  A) Donchian plain (champion)
  B) Donchian crisis-filtered (position zeroed on forward-flagged crisis)
across FULL, TRAIN, TEST -- so we see if the filter helps out-of-sample,
not just on the fit it was chosen from.

Honesty guards:
  - Crisis flag is FORWARD-FILTERED (predict_proba, regime_t uses data
    <= t). The filter cannot know a crisis before it is detectable.
  - HMM is fit on TRAIN only for the walk-forward columns; the TEST flag
    comes from applying the train-fit model forward. No test data trains
    the detector.
  - Per-instrument (CL first). No universal claim.

Run:  python -m runners.run_donchian_crisis_filter
"""

import numpy as np
import pandas as pd

from core.engine import run_engine
from core.load_panels import load_panels
from strategies.strategies_donchian_champions import (
    donchian_ensemble_macd_4step_take)
from diagnostics.hmm_regime_v2 import (
    make_features, fit_hmm, label_states)


COST = 0.001
TRAIN = ("2020-01-01", "2023-01-01")
TEST = ("2023-01-01", "2025-01-01")
FULL_START = "2020-07-01"


def crisis_flag(px, fit_px=None):
    """Forward-filtered boolean: True where regime_t == crisis.

    Args:
        px: price series to LABEL (produce flags for).
        fit_px: price series to FIT the HMM on. If None, fit on px itself
            (full-sample mode). For walk-forward, fit on TRAIN, label the
            TEST slice with the train-fit model.

    Returns:
        Series[bool] aligned to px's feature index (True = crisis).
    """
    feat_label = make_features(px)
    if fit_px is None:
        model = fit_hmm(feat_label)
        labels = label_states(model)
        post = model.predict_proba(feat_label.values)
    else:
        feat_fit = make_features(fit_px)
        model = fit_hmm(feat_fit)
        labels = label_states(model)
        post = model.predict_proba(feat_label.values)   # forward on label set
    crisis_state = [s for s, l in labels.items() if l == "crisis"][0]
    seq = post.argmax(axis=1)
    return pd.Series(seq == crisis_state, index=feat_label.index)


def apply_filter(pos, is_crisis):
    """Zero the position on crisis days (go to cash)."""
    is_crisis = is_crisis.reindex(pos.index).fillna(False).astype(bool)
    return pos.where(~is_crisis, other=0.0)


def run_pair(px, trade_start, fit_px=None):
    """Return (plain_ret, plain_dd, filt_ret, filt_dd, crisis_share)."""
    pos = donchian_ensemble_macd_4step_take(px)
    _, pr, pdd = run_engine(px, pos, trade_start, COST)

    flag = crisis_flag(px, fit_px=fit_px)
    pos_f = apply_filter(pos, flag)
    _, fr, fdd = run_engine(px, pos_f, trade_start, COST)

    share = flag.reindex(px.loc[trade_start:].index).fillna(False).mean()
    return pr, pdd, fr, fdd, share


def main():
    P = load_panels()
    SYM = "CL"
    px = P["close"][SYM].where(P["native"][SYM]).dropna()
    print("=" * 66)
    print(f"  {SYM}: plain Donchian vs crisis-filtered Donchian")
    print("=" * 66)

    def row(tag, pr, pdd, fr, fdd, share):
        d_ret = (fr - pr) * 100
        d_dd = (fdd - pdd) * 100
        print(f"  {tag:<16}"
              f"plain {pr*100:>+6.1f}%/{pdd*100:>5.1f}%   "
              f"filt {fr*100:>+6.1f}%/{fdd*100:>5.1f}%   "
              f"dRet {d_ret:>+5.1f}pp dDD {d_dd:>+5.1f}pp  "
              f"crisis {share*100:.0f}%")

    # FULL (fit on full history -- exploratory)
    r = run_pair(px, FULL_START, fit_px=None)
    row("FULL", *r)

    # WALK-FORWARD: fit HMM on TRAIN, apply forward to each slice
    tr = px[(px.index >= TRAIN[0]) & (px.index < TRAIN[1])]
    te = px[(px.index >= TEST[0]) & (px.index < TEST[1])]

    r_tr = run_pair(tr, tr.index[40], fit_px=tr)          # train fit on train
    row("TRAIN", *r_tr)
    r_te = run_pair(te, te.index[40], fit_px=tr)          # TEST labelled by TRAIN model
    row("TEST (tr-fit)", *r_te)

    print("\n  dRet>0 and dDD>0 (less negative) on TEST = filter helps")
    print("  out-of-sample: removes crisis losses without killing the")
    print("  trend/range profit. dRet<0 = filter cuts good trades too.")
    print("  crisis% = share of days flagged crisis (cash) in that window.")
    print("\n  Reminder: single instrument (CL). Next, sweep the same test")
    print("  across all 17 -- the filter may help some, hurt others. That")
    print("  per-instrument spread IS the finding, not a universal rule.")


if __name__ == "__main__":
    main()
