from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from .io import load_dataset_from_sheet
from .qa import qa_results_to_markdown, run_timeseries_qa


def load_registry(registry_path: Path) -> Dict[str, Any]:
    if not registry_path.exists():
        raise FileNotFoundError(f"registry.yaml not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as f:
        reg = yaml.safe_load(f) or {}
    if "datasets" not in reg:
        raise ValueError("registry.yaml must contain top-level key: datasets")
    return reg


def attach_metadata(long_df: pd.DataFrame, ds_meta: Dict[str, Any]) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    out = long_df.copy()
    out["frequency"] = ds_meta.get("frequency", "")
    out["unit"] = ds_meta.get("unit", "")
    out["notes"] = ds_meta.get("notes", "")
    return out


def run_data_ingestion(
    excel_path: Path,
    registry_path: Path,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Returns:
      - long_df_all: combined long-form time series
      - raw_tables: dataset -> raw/normalized DataFrame (for debugging)
    """
    reg = load_registry(registry_path)
    datasets: List[Dict[str, Any]] = reg["datasets"]

    all_long: List[pd.DataFrame] = []
    raw_tables: Dict[str, pd.DataFrame] = {}

    for ds in datasets:
        name = ds["dataset"]
        logging.info(f"[Task B] Loading dataset='{name}' sheet='{ds['sheet']}' kind='{ds['kind']}'")
        long_df, raw_df = load_dataset_from_sheet(excel_path=excel_path, ds=ds)
        raw_tables[name] = raw_df
        long_df = attach_metadata(long_df, ds)
        all_long.append(long_df)

    long_df_all = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()

    long_df_all["date"] = pd.to_datetime(long_df_all["date"], errors="coerce")
    long_df_all = long_df_all.dropna(subset=["date"])
    long_df_all["date"] = long_df_all["date"].dt.strftime("%Y-%m-%d")
    # write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)

    long_out_csv = output_dir / "processed" / "timeseries_long.csv"
    long_df_all.to_csv(long_out_csv, index=False, encoding="utf-8-sig")
    logging.info(f"[Task B] Wrote long-form timeseries to: {long_out_csv}")

    # QA
    qa_results = run_timeseries_qa(long_df_all)
    qa_md = qa_results_to_markdown(qa_results)
    qa_path = output_dir / "processed" / "QA_SUMMARY.md"
    qa_path.write_text(qa_md, encoding="utf-8")
    logging.info(f"[Task B] Wrote QA summary to: {qa_path}")

    return long_df_all, raw_tables
