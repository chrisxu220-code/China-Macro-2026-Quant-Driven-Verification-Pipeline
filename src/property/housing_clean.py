from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanSpec:
    price_xlsx_path: Path
    out_csv_path: Path
    header_row: int = 2  # 等价于你原来 read_excel(..., header=2)


def parse_sheet_name(name: str) -> tuple[str | None, str]:
    """
    Examples:
      202512新房
      202510_二手房
    """
    m = re.search(r"(\d{6})", name)
    date = f"{m.group(1)[:4]}-{m.group(1)[4:]}" if m else None

    house_type = "二手房" if "二手" in name else "新房"
    return date, house_type


def clean_one_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) 向下填充（处理合并单元格）
    df = df.ffill()

    all_parts: list[pd.DataFrame] = []

    # 左右两块（每块最多 4 列）
    for start_col in (0, 4):
        if start_col >= df.shape[1]:
            continue

        part = df.iloc[:, start_col : start_col + 4].copy()
        if part.shape[1] < 3:
            continue

        cols = ["city", "mom", "yoy"]
        if part.shape[1] >= 4:
            cols.append("ytd_avg")

        part = part.iloc[:, : len(cols)]
        part.columns = cols

        if "ytd_avg" not in part.columns:
            part["ytd_avg"] = np.nan

        all_parts.append(part)

    if not all_parts:
        return pd.DataFrame(columns=["city", "mom", "yoy", "ytd_avg"])

    df_all = pd.concat(all_parts, ignore_index=True)

    # 2) 城市名清理（保留你原来的逻辑）
    df_all["city"] = (
        df_all["city"]
        .astype(str)
        .str.strip()
        .str.replace(r"[\s\u3000]+", "", regex=True)
        .str.replace(r"市$", "", regex=True)
    )

    df_all = df_all[df_all["city"] != ""]
    df_all = df_all[~df_all["city"].isin(["nan", "None", "城市"])]
    df_all = df_all[df_all["city"].str.len() <= 4]

    # 3) 数值转 float
    for c in ["mom", "yoy", "ytd_avg"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # mom/yoy 都空：通常是表头垃圾行
    df_all = df_all.dropna(subset=["mom", "yoy"], how="all")

    return df_all


def build_panel(spec: CleanSpec) -> pd.DataFrame:
    xls = pd.ExcelFile(spec.price_xlsx_path)

    panel: list[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        date, house_type = parse_sheet_name(sheet)
        if date is None:
            continue

        df_raw = pd.read_excel(spec.price_xlsx_path, sheet_name=sheet, header=spec.header_row)
        df_clean = clean_one_sheet(df_raw)

        df_clean["date"] = date
        df_clean["house_type"] = house_type
        panel.append(df_clean)

    if not panel:
        raise ValueError("No valid sheets parsed. Check sheet names and header_row.")

    panel_df = pd.concat(panel, ignore_index=True)
    panel_df = panel_df[
        ["city", "date", "house_type", "mom", "yoy", "ytd_avg"]
    ].sort_values(["city", "date", "house_type"])

    return panel_df


def main():
    # 你可以先不接 config.yaml，先把路径改成你项目里的相对路径即可
    repo_root = Path(__file__).resolve().parents[2]  # .../src/property/housing_clean.py -> repo root
    default_xlsx = repo_root / "data" / "raw" / "housing price.xlsx"
    default_out = repo_root / "output" / "property" / "processed" / "housing_price_panel.csv"

    xlsx_path = Path(default_xlsx)
    out_csv = Path(default_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    spec = CleanSpec(price_xlsx_path=xlsx_path, out_csv_path=out_csv)

    df_panel = build_panel(spec)
    df_panel.to_csv(spec.out_csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote: {spec.out_csv_path}")


if __name__ == "__main__":
    main()
