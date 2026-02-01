from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .theme import COLORS, FONT_SETTINGS, apply_ib_style, annotate_latest
from src.utils.plotting import setup_mpl_chinese


setup_mpl_chinese()


@dataclass(frozen=True)
class ValidationSpec:
    sales_monthly_csv: Path                # output/property/housing_sales_clean_monthly.csv
    regime_table_csv: Path                 # output/property/monthly_regime_table.csv
    output_dir: Path = Path("output/property")
    figures_dir: Path = Path("output/property/figures")
    year: int | None = None
    lead_lags: tuple[int, ...] = (-6, -3, -1, 0, 1, 3, 6)


def _infer_year_from_path(p: Path) -> int | None:
    m = re.search(r"(20\d{2})", str(p))
    return int(m.group(1)) if m else None


def _infer_year_from_regime_table(reg: pd.DataFrame) -> int | None:
    if "date" not in reg.columns:
        return None
    dt = pd.to_datetime(reg["date"], errors="coerce").dropna()
    if dt.empty:
        return None
    return int(dt.max().year)


def _pick_house_type(reg: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    reg = reg.copy()
    # 强制转为字符串并做映射
    reg["house_type"] = reg["house_type"].astype(str)
    
    # 建立映射：把 csv 里的值映射回脚本认识的标签
    mapping = {"existing": "二手房", "new": "新房"}
    reg["house_type"] = reg["house_type"].replace(mapping)
    
    ht_available = reg["house_type"].unique().tolist()
    
    # 优先选新房
    target_ht = "新房" if "新房" in ht_available else ht_available[0]
    
    reg_sub = (
        reg.loc[reg["house_type"] == target_ht, ["date", "stabilization_prob"]]
        .dropna()
        .sort_values("date")
        .set_index("date")
    )
    return reg_sub, target_ht


def _leadlag_corr(prob: pd.Series, y: pd.Series, lags: list[int]) -> pd.DataFrame:
    """
    corr(prob[t], y[t+lag])
    +lag means y happens later => signal leads.
    """
    out = []
    x = pd.to_numeric(prob, errors="coerce")
    y0 = pd.to_numeric(y, errors="coerce")

    for k in lags:
        yy = y0.shift(-k)
        m = x.notna() & yy.notna()
        c = float(x[m].corr(yy[m])) if int(m.sum()) >= 4 else np.nan
        out.append({"series": str(y.name), "lag_months": int(k), "corr": c, "n": int(m.sum())})

    return pd.DataFrame(out)

def run_sales_inventory_validation(spec: ValidationSpec) -> Path:
    out_dir = Path(spec.output_dir)
    fig_dir = Path(spec.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- load inputs ---
    sales = pd.read_csv(spec.sales_monthly_csv)
    reg = pd.read_csv(spec.regime_table_csv)
    # 强制指定年份为 2025，防止匹配失败
    year = 2025 
    logging.info(f"[Task D3] Using fixed year={year} for validation alignment")
    # --- validate regime table ---
    if "date" not in reg.columns:
        raise KeyError(f"Regime table missing 'date'. cols={list(reg.columns)}")
    if "stabilization_prob" not in reg.columns:
        raise KeyError(f"Regime table missing 'stabilization_prob'. cols={list(reg.columns)}")

    reg["date"] = pd.to_datetime(reg["date"], errors="coerce")
    reg = reg.dropna(subset=["date"])

    logging.info(
        f"[Task D3] Loaded regime table: rows={len(reg)} cols={list(reg.columns)} path={spec.regime_table_csv}"
    )

    # --- choose house_type for validation ---
    reg_sub, target_ht = _pick_house_type(reg)
    if reg_sub.empty:
        raise ValueError(
            f"Regime table has no usable rows for house_type={target_ht}. "
            f"Check regime_table_csv={spec.regime_table_csv}."
        )

    # --- determine year for sales alignment ---
    year = spec.year
    if year is None:
        year = _infer_year_from_path(spec.sales_monthly_csv) or _infer_year_from_regime_table(reg)

    if year is None:
        dt_nonnull = pd.to_datetime(reg["date"], errors="coerce").notna().sum()
        raise ValueError(
            "Cannot infer year for sales alignment. "
            "Please set property.year in config.yaml (recommended). "
            f"Diagnostics: regime_rows={len(reg)}, regime_date_nonnull={dt_nonnull}, sales_csv={spec.sales_monthly_csv}"
        )

    logging.info(f"[Task D3] Using year={year} and target house_type={target_ht} for validation")

    # --- build sales date index from month + year ---
    required = {"month", "metric_key", "monthly_value"}
    missing = required - set(sales.columns)
    if missing:
        raise KeyError(
            f"Sales monthly csv missing required columns: {sorted(list(missing))}. cols={list(sales.columns)}"
        )

    sales = sales.copy()
    sales["month"] = pd.to_numeric(sales["month"], errors="coerce")
    sales = sales.dropna(subset=["month"])
    sales["month"] = sales["month"].astype(int)

    sales["date"] = pd.to_datetime(
        sales["month"].map(lambda m: f"{int(year)}-{m:02d}-01"),
        errors="coerce",
    )
    sales = sales.dropna(subset=["date"])

    # keep only single-month values (as per template)
    if "is_single_month" in sales.columns:
        sales = sales[sales["is_single_month"]].copy()

    # pivot wide
    sales_wide = (
        sales.pivot_table(
            index="date",
            columns="metric_key",
            values="monthly_value",
            aggfunc="first",
        )
        .sort_index()
    )

    # --- normalize likely column names to canonical ---
    def _find_col(keyword_list: list[str]) -> str | None:
        cols = list(sales_wide.columns)
        for c in cols:
            for kw in keyword_list:
                if kw in str(c):
                    return str(c)
        return None

    area_col = _find_col(["sales_area", "销售面积", "成交面积", "商品房销售面积"])
    value_col = _find_col(["sales_value", "销售额", "成交额", "商品房销售额"])
    inv_col = _find_col(["inventory", "待售", "库存", "商品房待售面积"])

    combo = sales_wide.copy()
    if area_col and "sales_area" not in combo.columns:
        combo = combo.rename(columns={area_col: "sales_area"})
    if value_col and "sales_value" not in combo.columns:
        combo = combo.rename(columns={value_col: "sales_value"})
    if inv_col and "inventory" not in combo.columns:
        combo = combo.rename(columns={inv_col: "inventory"})

    # merge regime
    combo = combo.merge(reg_sub, left_index=True, right_index=True, how="inner")
    combo = combo.sort_index()  # ✅ important: ensure chronological order for diff()

    if combo.empty:
        raise ValueError(
            "After aligning sales with regime table, got empty overlap. "
            f"sales_range=[{sales_wide.index.min()}..{sales_wide.index.max()}], "
            f"regime_range=[{reg_sub.index.min()}..{reg_sub.index.max()}]. "
            "Likely wrong year or sales month coverage mismatch."
        )
    
    # =========================================================
    # Lead/Lag correlation table + plot
    # =========================================================
    # =========================================================
    
    lags = list(spec.lead_lags)
    ll_tables = []
    for col in ["sales_area", "sales_value"]:
        if col in combo.columns:
            ll_tables.append(_leadlag_corr(combo["stabilization_prob"], combo[col], lags))

    if ll_tables:
        ll = pd.concat(ll_tables, ignore_index=True)
        
        # --- 暴力重置开始 ---
        plt.close('all') 
        fig, ax = plt.subplots(figsize=(10.5, 5), dpi=300)
        
        # 【物理级修复】强制清除该 ax 对象的任何单位记忆
        # 这是 test_debug_plot.py 成功而主程序失败的唯一区别
        ax.xaxis.units = None 
        
        # 核心：使用 .values 剥离所有元数据
        for series_name, g in ll.groupby("series"):
            g = g.sort_values("lag_months")
            ax.plot(g["lag_months"].values, g["corr"].values, 
                    marker="o", markersize=6, label=series_name, linewidth=2)
            
        apply_ib_style(ax, title="Cross-Correlation: Signal vs Sales", 
                       ylabel="Correlation", has_negative=True, is_timeseries=False)
        
        # 强力重刷格式
        from matplotlib.ticker import ScalarFormatter, FixedLocator
        ax.xaxis.set_major_locator(FixedLocator(lags)) 
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
        # 锁定范围，防止被“幽灵日期”带跑
        ax.set_xlim(min(lags) - 0.5, max(lags) + 0.5)
        ax.set_ylim(-1.05, 1.05)
        # --- 强效重置结束 ---
        
        ax.set_xlabel("Lag (months) | +lag = Sales Later (Signal Leads)", **FONT_SETTINGS["label"])
        ax.legend(frameon=False, loc="upper right", **FONT_SETTINGS["legend"])
        
        plt.tight_layout()
        plt.savefig(fig_dir / "validation_leadlag_corr.png", dpi=300, bbox_inches="tight")
        plt.close()

    # =========================================================
    # Figure 1: Sales activity vs stabilization probability
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # 绘图逻辑保持原样，仅注入专业颜色
    if "sales_area" in combo.columns:
        ax1.plot(combo.index, combo["sales_area"], color="#00204E", # Midnight Blue
                 label="Monthly Sales Area", linewidth=2)

    if "sales_value" in combo.columns:
        ax1.plot(combo.index, combo["sales_value"], color="#6A737B", # Slate Grey
                 label="Monthly Sales Value", linewidth=1.5, alpha=0.7)
    # 建立中文到英文的显示名映射
    DISPLAY_HT = {
        "新房": "New Homes",
        "二手房": "Existing Homes",
        "ALL": "National Aggregate"
    }
    
    # 转换当前的 target_ht 为英文显示名
    # 如果找不到映射，则保留原样（防止 hard code 导致报错）
    target_ht_en = DISPLAY_HT.get(target_ht, target_ht)
    apply_ib_style(ax1, 
                  title=f"Validation: Stabilization Signal vs Sales Activity ({target_ht_en})", 
                  ylabel="Sales Activity",
                  is_timeseries=True)
    
    # 强制每 2 个月显示一个刻度，防止标签挤在一起
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    # 副轴：稳定概率 (使用 Goldman Gold 强调信号)
    ax2 = ax1.twinx()
    ax2.plot(combo.index, combo["stabilization_prob"], color="#A68B5B", # Goldman Gold
             linestyle="--", label="Stabilization Prob (Signal)", linewidth=2.5)
    
    # 移除副轴冗余边框以保持简洁
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylabel("Probability (0–1)", **FONT_SETTINGS["label"])
    
    # 标注最新信号点
    annotate_latest(ax2, combo.reset_index(), "stabilization_prob", f"Latest {target_ht_en}")

    # 统一图例样式
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left", **FONT_SETTINGS["legend"])

    plt.tight_layout()
    plt.savefig(fig_dir / "validation_price_vs_sales.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================
    # Figure 2: Inventory destocking check (保持所有原始逻辑)
    # =========================================================
    if "inventory" in combo.columns:
        inv_raw = pd.to_numeric(combo["inventory"], errors="coerce")
        level_like = (inv_raw.abs().median() > 5000) or (inv_raw.abs().max() > 20000)

        if level_like:
            inv_delta = inv_raw.diff()
            if len(inv_delta) > 0: inv_delta.iloc[0] = pd.NA
            title_text = "Inventory Dynamics: Level Change (Δ < 0 = Destocking)"
        else:
            inv_delta = inv_raw.copy()
            if len(inv_delta) > 0: inv_delta.iloc[0] = pd.NA
            title_text = "Inventory Dynamics: Monthly Δ (Δ < 0 = Destocking)"

        # 原始逻辑：离群值处理与 Winzorize
        y_valid = inv_delta.dropna()
        if not y_valid.empty:
            med = y_valid.abs().median()
            if med > 0: inv_delta = inv_delta.mask(inv_delta.abs() > 20 * med)

            y_clean = inv_delta.dropna()
            if not y_clean.empty:
                inv_plot = inv_delta.clip(lower=y_clean.quantile(0.05), upper=y_clean.quantile(0.95))
                
                fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
                # 使用风险红（Alert Red: #9E1B32）来表示库存去化/累积
                ax.bar(combo.index, inv_plot, color="#9E1B32", alpha=0.6, label="Monthly Inventory Δ")
                
                apply_ib_style(ax, title=title_text, ylabel="Δ Inventory Area", has_negative=True)
                ax.legend(frameon=False, **FONT_SETTINGS["legend"])

                plt.tight_layout()
                plt.savefig(fig_dir / "inventory_destocking_check.png", dpi=300, bbox_inches="tight")
                plt.close()

    # =========================================================
    # Output table
    # =========================================================
    out_path = out_dir / "price_sales_inventory_validation_table.csv"
    combo.reset_index().to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info(f"[Task D3] Wrote validation table: {out_path}")

    return out_path
