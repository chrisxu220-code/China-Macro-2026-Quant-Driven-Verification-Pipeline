from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class InvestmentSpec:
    processed_long_csv: Path
    output_dir: Path = Path("output/domestic")
    figures_dir: Path = Path("output/domestic/figures")
    series: List[str] = None
    method: str = "zscore_mean"


def _load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
        raise ValueError(f"No matching series found for investment: {series_list}")

    wide = sub.pivot_table(index="date", columns="key", values="value", aggfunc="mean").sort_index()

    if method == "zscore_mean":
        z = wide.apply(_zscore, axis=0)
        idx = z.mean(axis=1, skipna=True)
    elif method == "level_mean":
        idx = wide.mean(axis=1, skipna=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    return pd.DataFrame({"date": wide.index, "investment_index": idx}).reset_index(drop=True)


def _plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    # 引入你已经改好的专业样式组件
    from .theme import COLORS, FONT_SETTINGS, apply_ib_style, annotate_latest

    # 1. 提升分辨率至 300 DPI 以符合研报标准
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    
    # 2. 强制使用“深夜蓝”作为投资指数的主色调
    ax.plot(df["date"], df["investment_index"], 
            color=COLORS["primary"], linewidth=2.5, label="Investment Index")
    
    # 3. 应用你定义的投行样式模板
    apply_ib_style(ax, title=title, ylabel="Index Level (z-score)", has_negative=True)
    
    # 4. 标注最新的指数值
    annotate_latest(ax, df, "investment_index", "Latest Index")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def run_investment(spec: InvestmentSpec) -> Path:
    logging.info("[Task F] Building investment demand proxy index...")

    if not spec.series:
        raise ValueError("InvestmentSpec.series is required (set in config.yaml)")

    df_long = _load_long(spec.processed_long_csv)
    idx = _make_index(df_long, spec.series, spec.method)

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.figures_dir.mkdir(parents=True, exist_ok=True)

    out_csv = spec.output_dir / "investment_index.csv"
    idx.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _plot(idx, spec.figures_dir / "investment_index.png", "Investment proxy index")

    logging.info(f"[Task F] Wrote: {out_csv}")
    return out_csv
