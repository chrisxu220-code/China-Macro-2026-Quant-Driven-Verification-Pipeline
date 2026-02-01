from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config / Spec
# -----------------------------
@dataclass(frozen=True)
class PriceWorkbookSpec:
    """
    70城价格指数工作簿（你这个 housing price.xlsx）
    sheet 命名形如：202512新房 / 202512二手房
    """
    excel_path: Path
    output_dir: Path

    # 统计阈值（按“指数=100”为中性）
    low_tail_threshold: float = 95.0   # <=95 视为“显著下行”
    neutral_threshold: float = 100.0   # >=100 视为“同比不跌/走平以上”

    # 使用哪个口径做分布（默认：yoy_index）
    distribution_field: str = "yoy_index"


# -----------------------------
# Helpers
# -----------------------------
_SHEET_RE = re.compile(r"^(?P<yyyymm>\d{6})(?P<market>新房|二手房)$")


def _normalize_city(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    x = str(s)
    # 去掉全角空格/各种空白
    x = re.sub(r"[\s\u3000]+", "", x)
    return x


def _to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _sheet_to_date_market(sheet_name: str) -> Optional[Tuple[pd.Timestamp, str]]:
    m = _SHEET_RE.match(sheet_name.strip())
    if not m:
        return None
    yyyymm = m.group("yyyymm")
    market_cn = m.group("market")
    dt = pd.to_datetime(yyyymm + "01", format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return None
    market = "new" if market_cn == "新房" else "existing"
    return dt, market


def _parse_single_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Match behavior of user's verified cleaning script:
    - df_raw is read with header=2
    - ffill merged cells
    - split into left/right blocks of 4 cols
    - city/mom/yoy/ytd_avg parsing
    """
    df = df_raw.copy()

    # 1) fill merged cells
    df = df.ffill()

    all_parts = []

    for start_col in [0, 4]:
        part = df.iloc[:, start_col:start_col + 4].copy()

        if part.shape[1] < 3:
            continue

        cols = ["city", "mom_index", "yoy_index"]
        if part.shape[1] >= 4:
            cols.append("ytd_index")

        part = part.iloc[:, :len(cols)]
        part.columns = cols

        if "ytd_index" not in part.columns:
            part["ytd_index"] = np.nan

        all_parts.append(part)

    df_all = pd.concat(all_parts, ignore_index=True)

    # 2) clean weird rows (same as your verified script)
    df_all = df_all[df_all["city"].notna()]
    df_all["city"] = (
        df_all["city"]
        .astype(str)
        .str.replace("\u3000", "", regex=False)  # 全角空格
        .str.replace("\xa0", "", regex=False)    # NBSP
        .str.replace(" ", "", regex=False)       # 你文件里这种怪空格（U+2002）
        .str.strip()
    )

    # drop empty city after cleaning
    df_all = df_all[df_all["city"].ne("")]
    df_all = df_all[df_all["city"].astype(str).str.len() <= 4]

    # 3) numeric conversion
    for c in ["mom_index", "yoy_index", "ytd_index"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    return df_all

def _mad(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    if x.empty:
        return np.nan
    med = x.median()
    return (x - med).abs().median()

def compute_distribution_metrics(
    panel: pd.DataFrame,
    distribution_field: str,
    low_tail_threshold: float,
    neutral_threshold: float,
) -> pd.DataFrame:
    """
    D2 output: one row per (date, market)
    - percentiles (p10/p50/p90)
    - MAD + IQR (dispersion)
    - tail shares (<=low_tail_threshold, >=neutral_threshold)
    - breadth / near-neutral mass
    """
    rows = []
    for (date, market), g in panel.groupby(["date", "market"], sort=True):
        x_idx = pd.to_numeric(g[distribution_field], errors="coerce")
        x = x_idx.dropna().astype(float)
        n = int(x.shape[0])

        if n == 0:
            rows.append(
                dict(
                    date=date, market=market, num_cities=0,
                    p10_idx=np.nan, p50_idx=np.nan, p90_idx=np.nan,
                    p10_pct=np.nan, p50_pct=np.nan, p90_pct=np.nan,
                    mad_idx=np.nan, mad_pct=np.nan,
                    iqr_idx=np.nan, iqr_pct=np.nan,
                    breadth_up=np.nan,
                    tail_down=np.nan,
                    tail_up=np.nan,
                    share_95_100=np.nan,
                    low_tail_threshold=low_tail_threshold,
                    neutral_threshold=neutral_threshold,
                    distribution_field=distribution_field,
                )
            )
            continue

        # percentiles on index level
        p10 = float(x.quantile(0.10))
        p50 = float(x.quantile(0.50))
        p90 = float(x.quantile(0.90))

        # dispersion
        mad = float(_mad(x))
        q25 = float(x.quantile(0.25))
        q75 = float(x.quantile(0.75))
        iqr = q75 - q25

        # shares
        tail_down = float((x <= low_tail_threshold).mean())
        tail_up = float((x >= neutral_threshold).mean())
        share_95_100 = float(((x >= low_tail_threshold) & (x < neutral_threshold)).mean())
        breadth_up = tail_up  # keep naming consistent with your csv

        # convert index->pct as (idx/100 - 1)
        def to_pct(v: float) -> float:
            return v / 100.0 - 1.0

        rows.append(
            dict(
                date=date,
                market=market,
                num_cities=n,

                p10_idx=p10,
                p50_idx=p50,
                p90_idx=p90,
                p10_pct=to_pct(p10),
                p50_pct=to_pct(p50),
                p90_pct=to_pct(p90),

                mad_idx=mad,
                mad_pct=mad / 100.0,      # MAD in “index points” scaled to pct-ish
                iqr_idx=iqr,
                iqr_pct=iqr / 100.0,

                breadth_up=breadth_up,
                tail_down=tail_down,
                tail_up=tail_up,
                share_95_100=share_95_100,

                low_tail_threshold=low_tail_threshold,
                neutral_threshold=neutral_threshold,
                distribution_field=distribution_field,
            )
        )

    return pd.DataFrame(rows).sort_values(["date", "market"]).reset_index(drop=True)


# -----------------------------
# Core: build city panel
# -----------------------------
def build_city_price_panel(excel_path: Path) -> pd.DataFrame:
    """
    读整个 workbook，拼成 long panel：
    date, market, city, mom_index, yoy_index, ytd_index
    """
    xls = pd.ExcelFile(excel_path)
    all_parts: List[pd.DataFrame] = []

    for sheet in xls.sheet_names:
        meta = _sheet_to_date_market(sheet)
        if meta is None:
            continue
        dt, market = meta
        df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=2)
        df_city = _parse_single_sheet(df_raw)
        part = _parse_single_sheet(df_raw)
        part.insert(0, "market", market)
        part.insert(0, "date", dt)
        all_parts.append(part)

    if not all_parts:
        raise ValueError(f"No valid sheets matched pattern YYYYMM新房/二手房 in: {excel_path}")

    panel = pd.concat(all_parts, ignore_index=True)

    # 强制类型
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    panel["market"] = panel["market"].astype(str)
    panel["city"] = panel["city"].astype(str)

    # 口径：把“指数”也转成“百分比变化”（以 100 为中性）
    # 例如 yoy_index=97.6 => yoy_pct = -0.024
    for f in ["mom_index", "yoy_index", "ytd_index"]:
        if f in panel.columns:
            panel[f] = pd.to_numeric(panel[f], errors="coerce")
            panel[f.replace("_index", "_pct")] = (panel[f] - 100.0) / 100.0

    # 去重（同 date/market/city 只保留一条，优先非空）
    panel = (
        panel.sort_values(["date", "market", "city"])
        .groupby(["date", "market", "city"], as_index=False)
        .agg(
            {
                "mom_index": "first",
                "yoy_index": "first",
                "ytd_index": "first",
                "mom_pct": "first",
                "yoy_pct": "first",
                "ytd_pct": "first",
            }
        )
    )

    return panel


# -----------------------------
# Core: distribution metrics
# -----------------------------
def _mad(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def compute_distribution_metrics(
    panel: pd.DataFrame,
    *,
    distribution_field: str = "yoy_index",
    low_tail_threshold: float = 95.0,
    neutral_threshold: float = 100.0,
) -> pd.DataFrame:
    """
    输出你要的：
    - breadth_up: share(index>=100)
    - tail_down: share(index<=95)
    - median, mad (基于 pct 版本更合理)
    - num_cities
    """
    if distribution_field not in panel.columns:
        raise ValueError(f"distribution_field='{distribution_field}' not in panel columns")

    # mad 用 pct 版本（更稳健）
    pct_field = distribution_field.replace("_index", "_pct")
    if pct_field not in panel.columns:
        # fallback：直接用 index 算 mad
        pct_field = distribution_field

    gcols = ["date", "market"]
    out_rows: List[Dict[str, object]] = []

    for (dt, market), g in panel.groupby(gcols):
        x = pd.to_numeric(g[distribution_field], errors="coerce").to_numpy()
        xp = pd.to_numeric(g[pct_field], errors="coerce").to_numpy()

        valid = x[~np.isnan(x)]
        validp = xp[~np.isnan(xp)]

        n = int(len(valid))
        if n == 0:
            out_rows.append(
                {
                    "date": dt,
                    "market": market,
                    "num_cities": 0,
                    "median_pct": np.nan,
                    "mad_pct": np.nan,
                    "breadth_up": np.nan,
                    "tail_down": np.nan,
                    "share_95_100": np.nan,
                }
            )
            continue

        breadth_up = float(np.mean(valid >= neutral_threshold))
        tail_down = float(np.mean(valid <= low_tail_threshold))
        share_95_100 = float(np.mean((valid > low_tail_threshold) & (valid < neutral_threshold)))

        median_pct = float(np.median(validp)) if len(validp) else np.nan
        mad_pct = _mad(validp) if len(validp) else np.nan

        out_rows.append(
            {
                "date": dt,
                "market": market,
                "num_cities": n,
                "median_pct": median_pct,
                "mad_pct": mad_pct,
                "breadth_up": breadth_up,
                "tail_down": tail_down,
                "share_95_100": share_95_100,
                "low_tail_threshold": low_tail_threshold,
                "neutral_threshold": neutral_threshold,
                "distribution_field": distribution_field,
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["market", "date"]).reset_index(drop=True)
    return out


# -----------------------------
# Entry: run D1
# -----------------------------
def run_property_price_distribution(spec: PriceWorkbookSpec) -> Tuple[Path, Path]:
    """
    运行 D1：产出 panel + metrics 两个 CSV
    """
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    panel = build_city_price_panel(spec.excel_path)
    panel_path = spec.output_dir / "price_city_panel.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")

    metrics = compute_distribution_metrics(
        panel=panel,
        distribution_field=spec.distribution_field,
        low_tail_threshold=spec.low_tail_threshold,
        neutral_threshold=spec.neutral_threshold,
    )
    metrics_path = spec.output_dir / "price_distribution_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    return panel_path, metrics_path
