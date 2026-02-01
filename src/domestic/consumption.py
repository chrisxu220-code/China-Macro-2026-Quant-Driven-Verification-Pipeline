from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ConsumptionSpec:
    processed_long_csv: Path                       # output/processed/timeseries_long.csv
    output_dir: Path = Path("output/domestic")
    figures_dir: Path = Path("output/domestic/figures")

    # series identifiers in your long table (exact match). Put them in config.
    series: List[str] = None  # e.g. ["retail_sales_yoy::社会消费品零售总额同比", "services_pmi::服务业PMI"]

    # method to aggregate multiple proxies into one index
    method: str = "zscore_mean"  # "zscore_mean" | "level_mean"


def _load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "series" not in df.columns or "value" not in df.columns:
        raise ValueError("processed_long_csv must have columns: date, series, value")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd


def _make_index(df_long: pd.DataFrame, series_list: List[str], method: str) -> pd.DataFrame:
    df_long["key"] = df_long["dataset"].astype(str) + "::" + df_long["series"].astype(str)
    sub = df_long[df_long["key"].isin(series_list)].copy()
    if sub.empty:
        raise ValueError(f"No matching series found for consumption: {series_list}")

    wide = sub.pivot_table(index="date", columns="key", values="value", aggfunc="mean").sort_index()

    if method == "zscore_mean":
        z = wide.apply(_zscore, axis=0)
        idx = z.mean(axis=1, skipna=True)
    elif method == "level_mean":
        idx = wide.mean(axis=1, skipna=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    out = pd.DataFrame({"date": wide.index, "consumption_index": idx}).reset_index(drop=True)
    return out


def _plot_index(df: pd.DataFrame, out_path: Path, title: str) -> None:
    # 引入你的投行样式组件
    from .theme import COLORS, FONT_SETTINGS, apply_ib_style, annotate_latest

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    
    # 使用深夜蓝绘制主趋势线
    ax.plot(df["date"], df.iloc[:, 1], color=COLORS["primary"], linewidth=2.5, label="Consumption Index")
    
    # 应用投行样式模板
    apply_ib_style(ax, title=title, ylabel="Index Level (z-score)", has_negative=True)
    
    # 标注最新数据点
    annotate_latest(ax, df, df.columns[1], "Latest Index")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def run_consumption(spec: ConsumptionSpec) -> Path:
    logging.info("[Task F] Building consumption demand proxy index...")

    if not spec.series:
        raise ValueError("ConsumptionSpec.series is required (set in config.yaml)")

    df_long = _load_long(spec.processed_long_csv)
    idx = _make_index(df_long, spec.series, spec.method)

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.figures_dir.mkdir(parents=True, exist_ok=True)

    out_csv = spec.output_dir / "consumption_index.csv"
    idx.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _plot_index(idx, spec.figures_dir / "consumption_index.png", "Consumption proxy index")

    logging.info(f"[Task F] Wrote: {out_csv}")
    return out_csv
