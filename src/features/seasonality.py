from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd


CNY_METHOD = Literal["merge_jan_feb", "none"]


@dataclass(frozen=True)
class CNYRule:
    """
    How to treat Jan/Feb distortion for a given series.
    - agg="sum": flows (e.g., TSF increment, loans, exports value)
    - agg="mean": rates/indices (e.g., PMI, PPI YoY, FX, growth rates)
    """
    series: str
    agg: Literal["sum", "mean"]


def _month_key(d: pd.Series) -> pd.Series:
    dt = pd.to_datetime(d, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def merge_jan_feb(
    df: pd.DataFrame,
    rules: Optional[list[CNYRule]] = None,
    keep_feb_as_anchor: bool = True,
) -> pd.DataFrame:
    """
    Merge Jan+Feb into a single point (Feb as anchor by default).
    Input: long-form df with columns: dataset, series, date, value (+ optional metadata)
    Output: same schema, but Jan removed and Feb replaced by merged value for configured series.

    If rules is None, we default:
      - if series name contains "增量" / "新增" / "出口" / "进口" -> sum
      - else -> mean
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    out["_m"] = out["date"].dt.month
    out["_y"] = out["date"].dt.year

    rule_map: dict[str, str] = {}
    if rules:
        rule_map = {r.series: r.agg for r in rules}

    def infer_agg(s: str) -> str:
        if s in rule_map:
            return rule_map[s]
        if any(k in s for k in ["增量", "新增", "出口", "进口", "融资", "贷款"]):
            return "sum"
        return "mean"

    # work per (dataset, series, year)
    keys = ["dataset", "series", "_y"]
    merged_rows = []

    g = out.groupby(keys, dropna=False)
    for (dataset, series, y), sub in g:
        jan = sub[sub["_m"] == 1]
        feb = sub[sub["_m"] == 2]

        if jan.empty or feb.empty:
            continue

        agg = infer_agg(str(series))
        val = None
        if agg == "sum":
            val = jan["value"].sum() + feb["value"].sum()
        else:
            val = pd.concat([jan["value"], feb["value"]]).mean()

        # anchor date
        anchor_date = feb["date"].max() if keep_feb_as_anchor else jan["date"].max()

        # take one representative row (keep metadata columns)
        base = feb.iloc[-1].copy()
        base["date"] = anchor_date
        base["value"] = float(val) if val is not None and not pd.isna(val) else np.nan
        merged_rows.append(base)

        # remove original Jan+Feb
        out.loc[jan.index, "_drop"] = True
        out.loc[feb.index, "_drop"] = True

    if merged_rows:
        out["_drop"] = out.get("_drop", False).fillna(False)
        out = out[out["_drop"] == False].copy()  # noqa: E712
        out = pd.concat([out, pd.DataFrame(merged_rows)], ignore_index=True)

    out = out.drop(columns=[c for c in ["_m", "_y", "_drop"] if c in out.columns])
    out = out.sort_values(["dataset", "series", "date"]).reset_index(drop=True)
    return out


def apply_seasonality(
    df: pd.DataFrame,
    method: CNY_METHOD = "merge_jan_feb",
    rules: Optional[list[CNYRule]] = None,
) -> pd.DataFrame:
    if method == "none":
        return df.copy()
    if method == "merge_jan_feb":
        return merge_jan_feb(df, rules=rules)
    raise ValueError(f"Unknown seasonality method: {method}")
