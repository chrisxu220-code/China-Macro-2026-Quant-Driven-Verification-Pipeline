from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .seasonality import apply_seasonality
from .fiscal import FiscalFuseSpec, build_fiscal_fuse, lead_lag_corr
from .structural_shift import ParadigmShiftSpec, paradigm_shift_index


@dataclass(frozen=True)
class FeatureBuildConfig:
    seasonality_method: str = "merge_jan_feb"
    # de-dup rule for (dataset, series, date)
    dedup_method: str = "mean"  # mean | last


def load_long_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # date may already be YYYY-MM-DD strings; coerce safely
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def dedup_long(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Ensure uniqueness per (dataset, series, date).
    """
    keys = ["dataset", "series", "date"]
    before = len(df)
    df = df.dropna(subset=["value"])
    if method == "last":
        df = df.sort_values(keys).groupby(keys, as_index=False).tail(1)
    else:
        # default: mean
        df = df.groupby(keys, as_index=False)["value"].mean()

    after = len(df)
    if after != before:
        logging.info(f"[Task C] De-dup applied: {before} -> {after} rows")
    return df


def pivot_monthly_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long form to monthly wide matrix indexed by month start.
    """
    df = df_long.copy()
    # normalize to month-start timestamps
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    df["_col"] = df["dataset"].astype(str) + "::" + df["series"].astype(str)

    wide = df.pivot_table(
        index="date",
        columns="_col",
        values="value",
        aggfunc="mean",
    )

    # flatten columns name
    wide.columns = [str(c) for c in wide.columns]
    return wide


def run_feature_build(
    processed_long_csv: Path,
    output_dir: Path,
    cfg: FeatureBuildConfig,
    fiscal_spec: Optional[FiscalFuseSpec] = None,
    shift_spec: Optional[ParadigmShiftSpec] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"[Task C] Loading long timeseries: {processed_long_csv}")
    df_long = load_long_timeseries(processed_long_csv)

    df_long = dedup_long(df_long, method=cfg.dedup_method)

    # seasonality (CNY distortion)
    logging.info(f"[Task C] Applying seasonality method: {cfg.seasonality_method}")
    df_long_adj = apply_seasonality(df_long, method=cfg.seasonality_method)

    # save adjusted long
    adj_path = output_dir / "timeseries_long_adjusted.csv"
    df_long_adj.to_csv(adj_path, index=False, encoding="utf-8-sig")
    logging.info(f"[Task C] Wrote adjusted long: {adj_path}")

    # pivot to wide monthly
    wide = pivot_monthly_wide(df_long_adj)
    wide_path = output_dir / "timeseries_wide_monthly.csv"
    wide.to_csv(wide_path, encoding="utf-8-sig")
    logging.info(f"[Task C] Wrote wide monthly: {wide_path}")

    # fiscal fuse (optional if spec provided)
    if fiscal_spec is not None:
        logging.info("[Task C] Building Fiscal Fuse features...")
        fuse = build_fiscal_fuse(wide, fiscal_spec)
        fuse_path = output_dir / "feature_fiscal_fuse.csv"
        fuse.to_csv(fuse_path, encoding="utf-8-sig")
        logging.info(f"[Task C] Wrote fiscal fuse: {fuse_path}")

        # also produce lead-lag corr table
        corr = lead_lag_corr(wide, x=fiscal_spec.tsf_series, y=fiscal_spec.target_series, max_lag=18)
        corr_path = output_dir / "diagnostic_fiscal_leadlag_corr.csv"
        corr.to_csv(corr_path, index=False, encoding="utf-8-sig")
        logging.info(f"[Task C] Wrote lead-lag corr: {corr_path}")
    else:
        logging.info("[Task C] Fiscal Fuse skipped (no fiscal_spec in config).")

    # paradigm shift index (optional if spec provided)
    if shift_spec is not None:
        logging.info("[Task C] Building Paradigm Shift index...")
        shift = paradigm_shift_index(wide, shift_spec)
        shift_path = output_dir / "feature_paradigm_shift.csv"
        shift.to_csv(shift_path, encoding="utf-8-sig")
        logging.info(f"[Task C] Wrote paradigm shift: {shift_path}")
    else:
        logging.info("[Task C] Paradigm Shift skipped (no shift_spec in config).")

    logging.info("[Task C] Feature build finished ✅")
