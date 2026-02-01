from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd


# ------------------------------------------------------------
# Housing sales workbook → clean monthly panel
# ------------------------------------------------------------
# Workbook shape assumed (matches housing_sales.xlsx):
# - first column: metric names (Chinese)
# - other columns: months as ints/strings (e.g., 12, 11, ..., 2)
# - values: cumulative (YTD) up to that month
#
# Output:
# - metric_key: canonical metric id
# - month: int (2..12)
# - cum_value: cumulative value up to that month
# - monthly_value: single-month value (diff of cumulative)
# - is_single_month: True for month>=3 (since month 2 is Jan+Feb cumulative in this template)
# ------------------------------------------------------------


DEFAULT_METRIC_MAP = {
    "房地产开发投资（亿元）": "investment",
    "新建商品房销售面积（万平方米）": "sales_area",
    "新建商品房销售额（亿元）": "sales_value",
    "商品房待售面积（万平方米）": "inventory",
}


@dataclass(frozen=True)
class SalesWorkbookSpec:
    excel_path: Path
    output_dir: Path = Path("output/property")
    year: int | None = None
    metric_map: dict[str, str] = None  # type: ignore[assignment]

    def resolved_metric_map(self) -> dict[str, str]:
        return self.metric_map or DEFAULT_METRIC_MAP


def _infer_year_from_path(p: Path) -> int | None:
    m = re.search(r"(20\d{2})", p.stem)
    return int(m.group(1)) if m else None


def build_sales_monthly_panel(excel_path: Path, metric_map: dict[str, str]) -> pd.DataFrame:
    df_raw = pd.read_excel(excel_path)

    # first col is metric name
    metric_col = df_raw.columns[0]
    month_cols = list(df_raw.columns[1:])

    # normalize month columns to int
    def parse_month(m) -> int:
        # handles int columns, '12', '12月', etc.
        s = str(m).strip()
        s = re.sub(r"\D", "", s)  # keep digits only
        if not s:
            raise ValueError(f"Unrecognized month column: {m!r}")
        return int(s)

    df_raw = df_raw.rename(columns={metric_col: "metric"})
    df_raw = df_raw.rename(columns={m: parse_month(m) for m in month_cols})

    df_long = (
        df_raw.melt(id_vars="metric", var_name="month", value_name="cum_value")
        .dropna(subset=["cum_value"])
    )
    df_long["month"] = df_long["month"].astype(int)

    # sort by month (input is usually 12→2)
    df_long = df_long.sort_values(["metric", "month"], ascending=[True, True])

    # cumulative → single month (diff)
    df_long["monthly_value"] = df_long.groupby("metric")["cum_value"].diff()
    df_long.loc[df_long["month"] == 2, "monthly_value"] = df_long.loc[df_long["month"] == 2, "cum_value"]
    df_long["is_single_month"] = df_long["month"] >= 2

    # keep only metrics you care about
    df_long = df_long[df_long["metric"].isin(metric_map)].copy()
    df_long["metric_key"] = df_long["metric"].map(metric_map)

    out = df_long[["metric_key", "month", "cum_value", "monthly_value", "is_single_month"]].sort_values(
        ["metric_key", "month"]
    )
    return out.reset_index(drop=True)


def run_property_sales_panel(spec: SalesWorkbookSpec) -> Path:
    metric_map = spec.resolved_metric_map()
    panel = build_sales_monthly_panel(spec.excel_path, metric_map)

    out_dir = spec.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "housing_sales_clean_monthly.csv"
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")

    # OPTIONAL: also write a wide monthly series
    year = spec.year or _infer_year_from_path(spec.excel_path)
    if year is not None:
        wide = sales_panel_to_wide(panel, year=year)
        wide_path = out_dir / "housing_sales_monthly_wide.csv"
        wide.to_csv(wide_path, encoding="utf-8-sig")

    return out_path



def sales_panel_to_wide(panel: pd.DataFrame, *, year: int) -> pd.DataFrame:
    """Convert long panel to wide monthly series with a proper datetime index."""
    df = panel.copy()
    df = df[df["is_single_month"]].copy()
    df["date"] = pd.to_datetime(df["month"].astype(int).map(lambda m: f"{year}-{m:02d}-01"))
    wide = (
        df.pivot_table(index="date", columns="metric_key", values="monthly_value", aggfunc="first")
        .sort_index()
    )
    wide.columns.name = None
    return wide
