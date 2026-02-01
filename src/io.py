from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _clean_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_unique_columns(cols: List[str]) -> List[str]:
    """
    Make column names unique by appending suffixes __2, __3, ...
    Only changes in-memory DataFrame columns; does NOT touch the Excel file.
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        base = c if c else "col"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
    return out

def _looks_like_date(x: Any) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return True
    # Excel serial date often shows as int/float like 45962
    if isinstance(x, (int, float)) and 30000 < float(x) < 60000:
        return True
    return False


def _to_datetime_safe(x: Any) -> Optional[pd.Timestamp]:
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, (np.datetime64,)):
        return pd.to_datetime(x)
    if isinstance(x, (int, float)) and 30000 < float(x) < 60000:
        # Excel serial date (days since 1899-12-30)
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(int(x), unit="D")
    try:
        return pd.to_datetime(x)
    except Exception:
        return None


def infer_data_start_row(df_raw: pd.DataFrame, lookahead: int = 15) -> int:
    """
    Heuristic: find the first row where the first column looks like a date.
    """
    for i in range(min(lookahead, len(df_raw))):
        v = df_raw.iloc[i, 0] if df_raw.shape[1] > 0 else None
        if _looks_like_date(v):
            return i
    # fallback: if cannot find, assume row 0 is header and data starts at 1
    return 1


def build_columns_from_header_rows(header_rows: pd.DataFrame) -> List[str]:
    """
    Combine multi-row headers by forward-filling and concatenating with '_'.
    """
    hr = header_rows.copy()
    hr = hr.applymap(_clean_text)
    hr = hr.replace("", np.nan).ffill(axis=1).fillna("")
    cols: List[str] = []
    for j in range(hr.shape[1]):
        parts = [p for p in hr.iloc[:, j].tolist() if p]
        col = "_".join(parts).strip("_")
        col = re.sub(r"_+", "_", col)
        cols.append(col if col else f"col_{j}")
    cols = ensure_unique_columns(cols)
    return cols


def normalize_simple_table(df_raw: pd.DataFrame, date_col_hint: Optional[str] = None) -> pd.DataFrame:
    """
    For sheets that already have a single header row (or near-single).
    """
    # find first non-empty row as header
    header_idx = 0
    for i in range(min(10, len(df_raw))):
        row = df_raw.iloc[i].tolist()
        if any(_clean_text(x) for x in row):
            header_idx = i
            break

    df = df_raw.copy()
    cols = [(_clean_text(c) or f"col_{i}") for i, c in enumerate(df.iloc[header_idx])]
    df.columns = ensure_unique_columns(cols)
    df = df.iloc[header_idx + 1 :].reset_index(drop=True)

    # drop fully empty rows
    df = df.dropna(how="all")
    return df


def normalize_multiheader_table(df_raw: pd.DataFrame, date_col_hint: Optional[str] = None) -> pd.DataFrame:
    """
    For sheets with multi-row headers; we detect data start and use rows above as header rows.
    """
    data_start = infer_data_start_row(df_raw)
    header_rows = df_raw.iloc[:data_start, :]
    cols = build_columns_from_header_rows(header_rows)

    df = df_raw.iloc[data_start:, :].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def to_long_timeseries(
    df: pd.DataFrame,
    dataset: str,
    date_col: str,
    value_cols: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert a wide table to long format with columns:
    dataset, series, date, value
    """
    if date_col not in df.columns:
        raise ValueError(f"[{dataset}] date_col '{date_col}' not found. columns={list(df.columns)[:10]}...")

    ddf = df.copy()
    date_obj = ddf[date_col]
    # If duplicate column names exist, pandas returns a DataFrame instead of Series.
    if isinstance(date_obj, pd.DataFrame):
    # Take the first column as the canonical date column
        date_obj = date_obj.iloc[:, 0]

    ddf[date_col] = date_obj.apply(_to_datetime_safe)
    ddf = ddf.dropna(subset=[date_col])

    if value_cols is None:
        value_cols = [c for c in ddf.columns if c != date_col]
    if id_cols is None:
        id_cols = [date_col]

    long_df = ddf.melt(id_vars=id_cols, value_vars=value_cols, var_name="series", value_name="value")
    long_df = long_df.rename(columns={date_col: "date"})
    long_df.insert(0, "dataset", dataset)

    # coerce numeric where possible
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])
    return long_df[["dataset", "series", "date", "value"]]


def parse_wide_indicator_dates(df_raw: pd.DataFrame, dataset: str, indicator_col: str = "指标") -> pd.DataFrame:
    """
    For sheets like 固投月度增长: rows are indicators, columns are dates.
    """
    # header row is row 0
    df = df_raw.copy()
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)
    df = df.dropna(how="all")
    # rename first col
    df = df.rename(columns={df.columns[0]: indicator_col})

    # melt: columns after indicator_col are dates
    date_cols = [c for c in df.columns if c != indicator_col]
    # ensure dates parseable
    melted = df.melt(id_vars=[indicator_col], value_vars=date_cols, var_name="date", value_name="value")
    melted["date"] = melted["date"].apply(_to_datetime_safe)
    melted = melted.dropna(subset=["date"])
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value"])

    melted.insert(0, "dataset", dataset)
    melted = melted.rename(columns={indicator_col: "series"})
    return melted[["dataset", "series", "date", "value"]]


def parse_new_home_price_index(df_raw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    For 新房价指数: date appears only on the first city row of each month.
    Need forward-fill dates.
    """
    # Try to locate header rows: first row has 日期/城市/新建...; second row has 环比/同比/定基...
    # We'll set columns by combining first two rows.
    header_rows = df_raw.iloc[:2, :].copy()
    cols = build_columns_from_header_rows(header_rows)

    df = df_raw.iloc[2:, :].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)

    # identify date & city columns
    date_col = None
    city_col = None
    for c in df.columns:
        if "日期" in c:
            date_col = c
        if c == "城市" or "城市" in c:
            city_col = c
    if date_col is None or city_col is None:
        raise ValueError(f"[{dataset}] cannot find date/city columns in 新房价指数. cols={df.columns.tolist()}")

    # forward fill date
    df[date_col] = df[date_col].ffill()
    df["date"] = df[date_col].apply(_to_datetime_safe)
    df = df.dropna(subset=["date"])
    df["city"] = df[city_col].astype(str)

    # choose numeric columns excluding date/city
    value_cols = [c for c in df.columns if c not in [date_col, city_col, "date", "city"]]
    out = df.melt(id_vars=["date", "city"], value_vars=value_cols, var_name="series", value_name="value")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])

    # include city in series name to avoid collisions
    out["series"] = out["city"] + " | " + out["series"]

    out.insert(0, "dataset", dataset)
    return out[["dataset", "series", "date", "value"]]


def load_sheet(excel_path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(excel_path, sheet_name=sheet, header=None)


def load_dataset_from_sheet(
    excel_path: Path,
    ds: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - long_df: standardized long-form time series (may be empty for raw_table)
      - raw_df: normalized table for debugging/inspection
    """
    dataset = ds["dataset"]
    sheet = ds["sheet"]
    kind = ds["kind"]
    date_hint = ds.get("date_col_hint")

    df_raw = load_sheet(excel_path, sheet)

    if kind == "simple_table":
        raw = normalize_simple_table(df_raw, date_hint)
        # date column: find first column containing hint, otherwise first column
        date_col = None
        if date_hint:
            for c in raw.columns:
                if date_hint in str(c):
                    date_col = c
                    break
        date_col = date_col or raw.columns[0]
        long_df = to_long_timeseries(raw, dataset=dataset, date_col=date_col)
        return long_df, raw

    if kind == "multiheader_table":
        raw = normalize_multiheader_table(df_raw, date_hint)
        date_col = raw.columns[0]
        long_df = to_long_timeseries(raw, dataset=dataset, date_col=date_col)
        return long_df, raw

    if kind == "wide_indicator_dates":
        raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
        long_df = parse_wide_indicator_dates(raw, dataset=dataset, indicator_col=ds.get("indicator_col", "指标"))
        return long_df, raw

    if kind == "new_home_price_index":
        raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
        long_df = parse_new_home_price_index(raw, dataset=dataset)
        return long_df, raw

    if kind == "raw_table":
        # keep as is; long_df empty for now
        raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
        long_df = pd.DataFrame(columns=["dataset", "series", "date", "value"])
        return long_df, raw

    raise ValueError(f"Unknown kind='{kind}' for dataset='{dataset}'")
