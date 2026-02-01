from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .theme import COLORS, FONT_SETTINGS, apply_ib_style

# -----------------------------
# Spec
# -----------------------------
@dataclass(frozen=True)
class TradeSpec:
    excel_path: Path
    sheet: str = "出口"
    output_dir: Path = Path("output/external")
    figures_dir: Path = Path("output/external/figures")

    # pick which period to focus in headline charts
    headline_period: str = "nov"  # or "jan_nov"

    # You can optionally map some groups to DM/EM buckets
    # If not provided, we will keep the original row labels.
    bucket_map: Dict[str, str] | None = None

    # rows to drop (notes/blank etc)
    drop_rows_pattern: str = r"^\s*$"

# -----------------------------
# Labels
# -----------------------------
REGION_EN: Dict[str, str] = {
    "总值": "Total",
    "美国": "United States",
    "东南亚国家联盟": "ASEAN",
    "欧洲联盟": "European Union",
    "亚太经济合作组织": "APEC",
    "区域全面经济伙伴关系协定（RCEP）成员国": "RCEP members",
    "共建“一带一路”国家和地区": "Belt and Road countries & regions",
}


PERIOD_KEY = {
    "11月": "nov",
    "1至11月": "jan_nov",
}
PERIOD_LABEL = {
    "nov": "Nov",
    "jan_nov": "Jan–Nov",
}
# -----------------------------
# Helpers
# -----------------------------
def _to_num(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s in {"-", "—", "–", ""}:
        return np.nan
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _find_header_rows(df_raw: pd.DataFrame) -> Tuple[int, int]:
    """
    We read sheet with header=None. Need to locate:
    - row_year: where year integers (2025, 2024) appear
    - row_period: where '11月' and '1至11月' appear
    Based on your preview, year appears in the first "header" row, period appears a few rows below.
    We'll locate robustly.
    """
    row_year = None
    row_period = None

    for i in range(min(len(df_raw), 20)):
        row = df_raw.iloc[i].tolist()
        # year row: contains at least one 20xx int-like
        years = []
        for v in row:
            if isinstance(v, (int, np.integer)) and 2000 <= int(v) <= 2100:
                years.append(int(v))
            else:
                s = _safe_str(v)
                if re.fullmatch(r"20\d{2}", s):
                    years.append(int(s))
        if years and row_year is None:
            row_year = i

        # period row: contains both '11月' and '1至11月' somewhere
        row_str = [_safe_str(v) for v in row]
        if ("11月" in row_str) and ("1至11月" in row_str) and row_period is None:
            row_period = i

    if row_year is None or row_period is None:
        raise ValueError("Could not locate year header row or period header row in sheet '出口'.")

    return row_year, row_period


def _extract_year_blocks(df_raw: pd.DataFrame, row_year: int) -> Dict[int, List[int]]:
    """
    Identify which columns belong to each year block by scanning row_year.
    In your file, year appears at col0 (2025) and col10 (2024).
    We'll treat each year marker as the start of a block until next year marker or end.
    """
    row = df_raw.iloc[row_year].tolist()
    year_pos = []
    for j, v in enumerate(row):
        if isinstance(v, (int, np.integer)) and 2000 <= int(v) <= 2100:
            year_pos.append((int(v), j))
        else:
            s = _safe_str(v)
            if re.fullmatch(r"20\d{2}", s):
                year_pos.append((int(s), j))

    if not year_pos:
        raise ValueError("No year markers found in the year header row.")

    year_pos = sorted(year_pos, key=lambda x: x[1])
    blocks: Dict[int, List[int]] = {}

    for idx, (yr, start) in enumerate(year_pos):
        end = year_pos[idx + 1][1] if idx + 1 < len(year_pos) else df_raw.shape[1]
        cols = list(range(start, end))
        blocks[yr] = cols

    return blocks


def _find_export_columns(
    df_raw: pd.DataFrame,
    year_cols: List[int],
    row_period: int
) -> Dict[str, int]:
    """
    Within a year block, find the column index for:
    - export_11m: column whose period header is '11月' and whose "category header" indicates 出口
    - export_ytd: period header '1至11月' and category 出口

    We don't fully rely on category row text, because the sheet is multiheader messy.
    Empirically in your preview:
      - export 11月 is at col3 inside 2025 block (global col3)
      - export 1至11月 is at col4
    We'll infer by looking for the *second pair* of ('11月','1至11月') inside block,
    because the first pair is usually 进出口, second is 出口, third is 进口.

    Strategy:
    - Get all columns in block whose period header is '11月' or '1至11月'
    - Group them by their relative order; pick the 2nd pair as export.
    """
    rel_cols = [c for c in year_cols if _safe_str(df_raw.iloc[row_period, c]) in {"11月", "1至11月"}]

    # build pairs in the order they appear
    pairs = []
    i = 0
    while i < len(rel_cols) - 1:
        c1, c2 = rel_cols[i], rel_cols[i + 1]
        p1 = _safe_str(df_raw.iloc[row_period, c1])
        p2 = _safe_str(df_raw.iloc[row_period, c2])
        if p1 == "11月" and p2 == "1至11月":
            pairs.append((c1, c2))
            i += 2
        else:
            i += 1

    if len(pairs) < 2:
        raise ValueError("Could not infer export columns: not enough (11月,1至11月) pairs found.")

    export_11m_col, export_ytd_col = pairs[1]  # 2nd pair => 出口
    return {
        "export_11m": export_11m_col,
        "export_ytd": export_ytd_col,
    }


def _parse_trade_sheet(excel_path: Path, sheet: str) -> Tuple[pd.DataFrame, Dict[int, Dict[str, int]]]:
    """
    Returns:
      trade_tidy: columns = [region, year, period, export_value]
      export_cols_by_year: {year: {"export_11m": col, "export_ytd": col}}
    """
    df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    row_year, row_period = _find_header_rows(df_raw)
    blocks = _extract_year_blocks(df_raw, row_year=row_year)

    export_cols_by_year: Dict[int, Dict[str, int]] = {}
    for yr, cols in blocks.items():
        export_cols_by_year[yr] = _find_export_columns(df_raw, cols, row_period=row_period)

    # Data rows start after row_period; find first row that has a non-empty region label
    start_row = row_period + 1
    # find region label column per year block: it's the first col of that block
    # For your file: 2025 block starts at col0, 2024 block starts at col10, but row labels live in col0.
    region_col = min([min(cols) for cols in blocks.values()])

    records = []
    for i in range(start_row, df_raw.shape[0]):
        region = _safe_str(df_raw.iloc[i, region_col])
        if not region or re.match(r"^\s*$", region):
            continue

        # keep only rows that look like region labels (not the header junk)
        # your first real data row is "总值"
        if region in {"单位：万元人民币"}:
            continue

        for yr in sorted(export_cols_by_year.keys(), reverse=True):
            cols = export_cols_by_year[yr]
            v11 = _to_num(df_raw.iloc[i, cols["export_11m"]])
            vytd = _to_num(df_raw.iloc[i, cols["export_ytd"]])
            records.append({"region": region, "year": yr, "period": "11月", "export_value": v11})
            records.append({"region": region, "year": yr, "period": "1至11月", "export_value": vytd})

    trade_tidy = pd.DataFrame.from_records(records)
    if trade_tidy.empty:
        raise ValueError("Parsed trade table is empty. Please verify sheet layout.")

    # Drop rows with all NaN exports across both periods
    trade_tidy = trade_tidy.dropna(subset=["export_value"], how="all")

    return trade_tidy, export_cols_by_year


def _build_two_point_panel(trade_tidy: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Build a panel comparing two years for a chosen period (11月 or 1至11月):
      region, value_2024, value_2025, delta, pct_change, share_2024, share_2025, share_delta
    """
    sub = trade_tidy[trade_tidy["period"] == period].copy()

    pivot = sub.pivot_table(index="region", columns="year", values="export_value", aggfunc="sum")
    # Expect 2024 & 2025
    for yr in [2024, 2025]:
        if yr not in pivot.columns:
            pivot[yr] = np.nan

    pivot = pivot.rename(columns={2024: "value_2024", 2025: "value_2025"}).reset_index()
    pivot["delta"] = pivot["value_2025"] - pivot["value_2024"]
    pivot["pct_change"] = pivot["delta"] / pivot["value_2024"]

    total_2024 = float(pivot.loc[pivot["region"] == "总值", "value_2024"].iloc[0]) if (pivot["region"] == "总值").any() else float(pivot["value_2024"].sum(skipna=True))
    total_2025 = float(pivot.loc[pivot["region"] == "总值", "value_2025"].iloc[0]) if (pivot["region"] == "总值").any() else float(pivot["value_2025"].sum(skipna=True))

    # Avoid divide-by-zero
    pivot["share_2024"] = pivot["value_2024"] / total_2024 if total_2024 else np.nan
    pivot["share_2025"] = pivot["value_2025"] / total_2025 if total_2025 else np.nan
    pivot["share_delta"] = pivot["share_2025"] - pivot["share_2024"]

    # sort: exclude 总值 from ranking lists
    pivot["_rank"] = np.where(pivot["region"] == "总值", -999, pivot["share_2025"].fillna(-1))
    pivot = pivot.sort_values(["_rank"], ascending=False).drop(columns=["_rank"])

    return pivot


def _apply_bucket(panel: pd.DataFrame, bucket_map: Dict[str, str]) -> pd.DataFrame:
    df = panel.copy()
    df["bucket"] = df["region"].map(bucket_map).fillna(df["region"])
    agg = df.groupby("bucket", as_index=False).agg(
        value_2024=("value_2024", "sum"),
        value_2025=("value_2025", "sum"),
        delta=("delta", "sum"),
    )
    agg["pct_change"] = agg["delta"] / agg["value_2024"]
    total_2024 = float(agg["value_2024"].sum(skipna=True))
    total_2025 = float(agg["value_2025"].sum(skipna=True))
    agg["share_2024"] = agg["value_2024"] / total_2024 if total_2024 else np.nan
    agg["share_2025"] = agg["value_2025"] / total_2025 if total_2025 else np.nan
    agg["share_delta"] = agg["share_2025"] - agg["share_2024"]
    return agg.sort_values("share_2025", ascending=False)

def _save_bar_compare(panel: pd.DataFrame, value_col_a: str, value_col_b: str, label_col: str, out_path: Path, title: str) -> None:
    df = panel[panel[label_col] != "总值"].copy()
    df = df.head(12)

    labels = df[label_col].tolist()
    a = df[value_col_a].astype(float).to_numpy()
    b = df[value_col_b].astype(float).to_numpy()

    y = np.arange(len(labels))
    h = 0.38

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(labels))), dpi=220)
    # 强制指定专业配色：2024用深夜蓝，2025用高盛金
    ax.barh(y - h/2, a, height=h, label="2024", color="#00204E", alpha=0.8)
    ax.barh(y + h/2, b, height=h, label="2025", color="#A68B5B")
    
    # 应用投行样式模板
    apply_ib_style(ax, title=title, ylabel="", is_timeseries=False)
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels(labels, **FONT_SETTINGS["label"])
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", **FONT_SETTINGS["legend"])
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def _save_share_change(panel: pd.DataFrame, label_col: str, out_path: Path, title: str) -> None:
    df = panel[panel[label_col] != "总值"].copy()
    df = df.sort_values("share_delta", ascending=False).head(12)

    labels = df[label_col].tolist()
    d = df["share_delta"].astype(float).to_numpy()

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(labels))), dpi=220)
    
    # 使用深夜蓝 (#00204E) 作为线条色，高盛金 (#A68B5B) 作为端点色
    ax.axvline(0, color="#6A737B", linewidth=1, alpha=0.5)
    ax.hlines(y, 0, d, color="#00204E", linewidth=1.5, alpha=0.6)
    ax.scatter(d, y, color="#A68B5B", s=60, zorder=3)
    
    apply_ib_style(ax, title=title, ylabel="", is_timeseries=False)
    
    from matplotlib.ticker import PercentFormatter
    ax.set_yticks(y)
    ax.set_yticklabels(labels, **FONT_SETTINGS["label"])
    ax.invert_yaxis()
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def _plot_share_bars_both_periods(trade_tidy: pd.DataFrame, out_path: Path, title: str) -> None:
    p11 = _build_two_point_panel(trade_tidy, "11月")
    pytd = _build_two_point_panel(trade_tidy, "1至11月")

    def prep(panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.copy()
        df = df[df["region"] != "总值"].copy()
        df["label"] = df["region"].map(REGION_EN).fillna(df["region"])
        # 选 Top12：按 2025 share 排序
        df = df.sort_values("share_2025", ascending=False).head(12)
        return df

    a = prep(p11)
    b = prep(pytd)

    # 让两张子图顺序一致：用 Jan–Nov 的排序做基准（更稳定）
    order = b["label"].tolist()
    a = a.set_index("label").reindex(order).reset_index()
    b = b.set_index("label").reindex(order).reset_index()

    def draw(ax, df: pd.DataFrame, subtitle: str) -> None:
        labels = df["label"].tolist()
        y = np.arange(len(labels))
        h = 0.38

        s24_raw = df["share_2024"].astype(float).to_numpy()
        s25_raw = df["share_2025"].astype(float).to_numpy()

        # draw bars: fill NaN (missing base-year) as 0 so bar still exists
        s24 = np.nan_to_num(s24_raw, nan=0.0)
        s25 = np.nan_to_num(s25_raw, nan=0.0)

        ax.barh(y - h/2, s24, height=h, label="2024", color="#00204E", alpha=0.85)
        ax.barh(y + h/2, s25, height=h, label="2025", color="#A68B5B")

        # 关键：显式关闭时间序列格式
        apply_ib_style(ax, title=subtitle, ylabel="", is_timeseries=False)
        
        # 确保 X 轴是普通数字
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks(y)
        ax.set_yticklabels(labels, **FONT_SETTINGS["label"])
        ax.invert_yaxis()
        
        xmax = (max(np.nanmax(s24), np.nanmax(s25)) * 1.18) if len(labels) else 1.0
        ax.set_xlim(0, xmax)
        x_col = xmax * 0.88  # Δpp 列位置
        
        # 数据标注 (保持原逻辑，优化字体大小)
        for i in range(len(labels)):
            # 2024 数据百分比
            if np.isfinite(s24_raw[i]):
                ax.text(s24[i], y[i] - h/2, f"  {s24_raw[i]*100:.1f}%", va="center", fontsize=8)
            else:
                ax.text(0.002, y[i] - h/2, "  n/a", va="center", fontsize=8)

            # 2025 数据百分比
            if np.isfinite(s25_raw[i]):
                ax.text(s25[i], y[i] + h/2, f"  {s25_raw[i]*100:.1f}%", va="center", fontsize=8)
            else:
                ax.text(0.002, y[i] + h/2, "  n/a", va="center", fontsize=8)

            # Δpp 标注 (右侧列)
            if np.isfinite(s24_raw[i]) and np.isfinite(s25_raw[i]):
                dpp = (s25_raw[i] - s24_raw[i]) * 100.0
                # 对于显著变化使用高盛金强调颜色
                text_color = "#A68B5B" if abs(dpp) > 0.5 else "#6A737B"
                ax.text(x_col, y[i], f"Δpp {dpp:+.1f}", va="center", ha="left", 
                        color=text_color, fontweight='bold', fontsize=8)
            else:
                ax.text(x_col, y[i], "Δpp new", va="center", ha="left", fontsize=8)
        
        # 列标题
        ax.text(x_col, -0.8, "Δpp (2025–2024)", ha="left", va="bottom", 
                fontsize=9, fontweight='bold', color="#6A737B")

    # --- 外部布局与保存逻辑 ---
    fig, axes = plt.subplots(2, 1, figsize=(14, max(7, 0.55 * len(order) * 2)), dpi=300)
    draw(axes[0], a, "Nov destination share (% of exports): 2024 vs 2025")
    draw(axes[1], b, "Jan–Nov destination share (% of exports): 2024 vs 2025")

    handles, labels_ = axes[0].get_legend_handles_labels()
    # 投行风格的大标题样式
    fig.suptitle(title, y=0.995, **FONT_SETTINGS["title"])
    fig.legend(handles, labels_, loc="upper center", ncol=2, 
               bbox_to_anchor=(0.5, 0.965), frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_trade(spec: TradeSpec) -> Dict[str, Path]:
    """
    Main entry:
      - parse trade sheet
      - build two-point panel for headline period (11月 or 1至11月)
      - write csv + figures
    Returns paths of outputs.
    """
    logging.info("[Task E] Running trade decomposition...")

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.figures_dir.mkdir(parents=True, exist_ok=True)

    trade_tidy, _ = _parse_trade_sheet(spec.excel_path, spec.sheet)
    trade_tidy = trade_tidy.copy()
    trade_tidy["period_key"] = trade_tidy["period"].map(PERIOD_KEY).fillna(trade_tidy["period"])
    trade_tidy["period_en"] = trade_tidy["period_key"].map(PERIOD_LABEL).fillna(trade_tidy["period"])
    trade_tidy["region_en"] = trade_tidy["region"].map(REGION_EN).fillna(trade_tidy["region"])
    tidy_path = spec.output_dir / "trade_tidy.csv"
    trade_tidy.to_csv(tidy_path, index=False, encoding="utf-8-sig")
    logging.info(f"[Task E] Wrote: {tidy_path}")

    periods_key = ["nov", "jan_nov"]

    outputs: Dict[str, Path] = {"trade_tidy": tidy_path}

    for per_key in periods_key:
        per_cn = "11月" if per_key == "nov" else "1至11月"
        panel = _build_two_point_panel(trade_tidy, period=per_cn)

        # attach english labels (preserve original chinese too)
        panel = panel.copy()
        panel["region_en"] = panel["region"].map(REGION_EN).fillna(panel["region"])

        panel_path = spec.output_dir / f"trade_two_point_panel_{per_key}.csv"
        panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
        outputs[f"trade_panel_{per_key}"] = panel_path

        # bucket (optional)
        bucket_panel = None
        bucket_path = None
        if spec.bucket_map:
            bucket_panel = _apply_bucket(panel, spec.bucket_map)
            bucket_panel = bucket_panel.copy()
            bucket_panel["bucket_en"] = bucket_panel["bucket"].map(REGION_EN).fillna(bucket_panel["bucket"])
            bucket_path = spec.output_dir / f"trade_bucket_panel_{per_key}.csv"
            bucket_panel.to_csv(bucket_path, index=False, encoding="utf-8-sig")
            outputs[f"trade_bucket_panel_{per_key}"] = bucket_path

        # Choose labels for plotting
        plot_df = bucket_panel if bucket_panel is not None else panel
        label_col = "bucket_en" if bucket_panel is not None else "region_en"

        # nicer charts
        fig1 = spec.figures_dir / f"export_value_compare_{per_key}.png"
        _save_bar_compare(
            plot_df,
            value_col_a="value_2024",
            value_col_b="value_2025",
            label_col=label_col,
            out_path=fig1,
            title=f"Exports by destination ({PERIOD_LABEL[per_key]}): 2024 vs 2025 (unit: RMB 10k)",
        )
        outputs[f"fig_value_compare_{per_key}"] = fig1

        fig2 = spec.figures_dir / f"export_share_delta_{per_key}.png"
        _save_share_change(
            plot_df,
            label_col=label_col,
            out_path=fig2,
            title=f"Change in destination share ({PERIOD_LABEL[per_key]}): 2025 - 2024",
        )
        outputs[f"fig_share_delta_{per_key}"] = fig2

        # Backward-compatible keys (headline period only) — MUST come AFTER variables exist
        if per_key == spec.headline_period:
            outputs["trade_panel"] = panel_path
            outputs["fig_value_compare"] = fig1
            outputs["fig_share_delta"] = fig2
            if bucket_panel is not None and bucket_path is not None:
                outputs["trade_bucket_panel"] = bucket_path


    # One combined slope figure with legend distinguishing periods
    # (Use non-bucket view by default to keep your original groups)
    fig3 = spec.figures_dir / "export_destination_share_nov_vs_jan_nov.png"
    _plot_share_bars_both_periods(
        trade_tidy,
        out_path=fig3,
        title="Destination share comparison: Nov vs Jan–Nov (2024 vs 2025)",
    )
    outputs["fig_share_bars_both_periods"] = fig3



    logging.info("[Task E] Trade decomposition finished ✅")
    return outputs
