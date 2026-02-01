from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


SCALE = Literal["zscore", "robust"]


@dataclass(frozen=True)
class ParadigmShiftSpec:
    new_engine_series: Sequence[str]      # e.g., FAI electronics/ICT, EV, semi...
    old_economy_series: Sequence[str]      # e.g., real estate/infrastructure/traditional manufacturing...
    scale: SCALE = "robust"
    composite: Literal["diff", "ratio"] = "diff"   # diff: new-old; ratio: new/old (after scaling)
    winsorize: float = 0.01


def _winsorize(s: pd.Series, p: float) -> pd.Series:
    if p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad + 1e-12)


def paradigm_shift_index(
    wide: pd.DataFrame,
    spec: ParadigmShiftSpec,
) -> pd.DataFrame:
    """
    Build a composite index capturing "new engines vs old economy".
    Returns df with:
      - new_comp
      - old_comp
      - paradigm_index
    """
    missing_new = [c for c in spec.new_engine_series if c not in wide.columns]
    missing_old = [c for c in spec.old_economy_series if c not in wide.columns]

    if missing_new or missing_old:
        preview = list(wide.columns)[:80]
        raise KeyError(
            "ParadigmShift: missing required series in wide columns.\n"
            f"missing_new={missing_new}\n"
            f"missing_old={missing_old}\n"
            f"available_columns_preview={preview}"
        )
    df = wide.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("wide must be indexed by DatetimeIndex.")

    missing = [c for c in list(spec.new_engine_series) + list(spec.old_economy_series) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing series in wide: {missing}")

    missing_new = [c for c in spec.new_engine_series if c not in df.columns]
    missing_old = [c for c in spec.old_economy_series if c not in df.columns]
    if missing_new or missing_old:
        preview = list(df.columns)[:80]
        raise KeyError(
            "ParadigmShift: missing required series in wide columns.\n"
            f"missing_new={missing_new}\n"
            f"missing_old={missing_old}\n"
            f"available_columns_preview={preview}"
        )

    new_mat = df[list(spec.new_engine_series)].apply(pd.to_numeric, errors="coerce")
    old_mat = df[list(spec.old_economy_series)].apply(pd.to_numeric, errors="coerce")

    # winsorize per column
    new_mat = new_mat.apply(lambda s: _winsorize(s, spec.winsorize))
    old_mat = old_mat.apply(lambda s: _winsorize(s, spec.winsorize))

    scaler = _robust_z if spec.scale == "robust" else _zscore

    new_scaled = new_mat.apply(scaler)
    old_scaled = old_mat.apply(scaler)

    new_comp = new_scaled.mean(axis=1)
    old_comp = old_scaled.mean(axis=1)

    out = pd.DataFrame(index=df.index)
    out["new_comp"] = new_comp
    out["old_comp"] = old_comp

    if spec.composite == "diff":
        out["paradigm_index"] = out["new_comp"] - out["old_comp"]
    else:
        out["paradigm_index"] = out["new_comp"] / (out["old_comp"].replace(0, np.nan))

    return out
