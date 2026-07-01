"""Load the preprocessed commodity panels (panel_*.parquet).

Thin loader so runners don't repeat the paths / alignment checks. Panels
were built by scripts/preprocess_basket.py: union calendar, ffilled
prices, carry on native dates only, volume zeroed on ffilled days, and a
boolean native mask. See that script for the guarantees.
"""

import pandas as pd

DATA = "/Users/shalygin/dev/Python_work/EMA1221/data/"


def load_panels(data_dir=DATA):
    """Load the four aligned commodity panels.

    Returns:
        dict with keys 'close', 'volume', 'rollyield', 'native' -- each a
        DataFrame on the same date index x 17 instrument columns.

    Raises:
        AssertionError if the panels are not index/column aligned, which
        would silently corrupt cross-sectional ranking.
    """
    close = pd.read_parquet(data_dir + "panel_close.parquet")
    volume = pd.read_parquet(data_dir + "panel_volume.parquet")
    rollyield = pd.read_parquet(data_dir + "panel_rollyield.parquet")
    native = pd.read_parquet(data_dir + "panel_native.parquet").astype(bool)

    for name, p in [("volume", volume), ("rollyield", rollyield),
                    ("native", native)]:
        assert p.index.equals(close.index), f"{name} index misaligned vs close"
        assert list(p.columns) == list(close.columns), \
            f"{name} columns misaligned vs close"

    return {"close": close, "volume": volume,
            "rollyield": rollyield, "native": native}
