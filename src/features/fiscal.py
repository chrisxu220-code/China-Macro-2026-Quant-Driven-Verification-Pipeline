from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FiscalFuseSpec:
    tsf_series: str                 # e.g., "社会融资规模增量统计__2" or a cleaned alias
    target_series: str              # e.g., industrial value added growth
    lead_months: int = 12
    zscore: bool = True


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def build_fiscal_fuse(
    wide: pd.DataFrame,
    spec: FiscalFuseSpec,
) -> pd.DataFrame:
    """
    wide: monthly wide df indexed by date, columns = series names, values numeric.
    output columns:
      - tsf
      - tsf_lead (tsf shifted by lead_months)
      - target
      - tsf_lead_z, target_z (optional)
    """
    df = wide.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("wide must be indexed by DatetimeIndex (monthly timestamps).")

    if spec.tsf_series not in df.columns:
        raise KeyError(f"Missing tsf_series='{spec.tsf_series}' in wide.columns")
    if spec.target_series not in df.columns:
        raise KeyError(f"Missing target_series='{spec.target_series}' in wide.columns")

    out = pd.DataFrame(index=df.index)
    out["tsf"] = pd.to_numeric(df[spec.tsf_series], errors="coerce")
    out["target"] = pd.to_numeric(df[spec.target_series], errors="coerce")

    out["tsf_lead"] = out["tsf"].shift(spec.lead_months)

    if spec.zscore:
        out["tsf_lead_z"] = _z(out["tsf_lead"])
        out["target_z"] = _z(out["target"])

    return out.dropna(how="all")


def lead_lag_corr(
    wide: pd.DataFrame,
    x: str,
    y: str,
    max_lag: int = 18,
) -> pd.DataFrame:
    """
    Compute corr(x shifted by lag, y) for lag in [-max_lag, +max_lag].
    Positive lag means x leads y (x shifted forward -> earlier x aligned with later y).
    """
    if x not in wide.columns or y not in wide.columns:
        raise KeyError(f"Missing columns: x='{x}' or y='{y}'")

    s_x = pd.to_numeric(wide[x], errors="coerce")
    s_y = pd.to_numeric(wide[y], errors="coerce")

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        corr = s_x.shift(lag).corr(s_y)
        rows.append({"lag_months": lag, "corr": float(corr) if corr is not None else np.nan})
    return pd.DataFrame(rows).sort_values("lag_months").reset_index(drop=True)
