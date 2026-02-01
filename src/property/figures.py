from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt
from .theme import COLORS, FONT_SETTINGS, apply_ib_style, annotate_latest
from src.utils.plotting import setup_mpl_chinese
setup_mpl_chinese()

# 局部映射，确保“新房/二手房”颜色统一
MARKET_COLORS = {
    "new": "#00204E",        # Midnight Blue
    "existing": "#A68B5B",   # Goldman Gold
    "新房": "#00204E",
    "二手房": "#A68B5B"
}

def _pick_quantile_cols(cols):
    """
    Pick an available (lo, mid, hi) quantile triple from metrics columns.
    Prefer 10/50/90, then 5/50/95, then 25/50/75.
    """
    candidates = [
        ("p10_pct", "p50_pct", "p90_pct"),
        ("p05_pct", "p50_pct", "p95_pct"),
        ("p25_pct", "p50_pct", "p75_pct"),
    ]
    for lo, mid, hi in candidates:
        if lo in cols and mid in cols and hi in cols:
            return lo, mid, hi
    return None, None, None


def make_property_figures(
    metrics_csv: Path,
    out_dir: Path = Path("output/property/figures"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(metrics_csv)

    # --- minimal schema checks ---
    required_base = ["date", "market"]
    missing = [c for c in required_base if c not in m.columns]
    if missing:
        raise KeyError(f"metrics_csv missing columns: {missing}. got={list(m.columns)}")

    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date"])

    for market, g in m.groupby("market"):
        g = g.sort_values("date").copy()

        # =========================

        # 1) Band chart [Style Refactor]
        # =========================
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        m_color = MARKET_COLORS.get(market.lower(), "#6A737B")
        
        lo_col, mid_col, hi_col = _pick_quantile_cols(g.columns)
        if lo_col is not None:
            ax.fill_between(g["date"], g[lo_col], g[hi_col], color=m_color, alpha=0.15)
            ax.plot(g["date"], g[mid_col], color=m_color, linewidth=2, label="Median YoY")
            title_band = f"({lo_col}-{hi_col})"
        else:
            # Fallback: median ± MAD
            lo = g["median_pct"] - g["mad_pct"]
            hi = g["median_pct"] + g["mad_pct"]
            ax.fill_between(g["date"], lo, hi, color=m_color, alpha=0.15)
            ax.plot(g["date"], g["median_pct"], color=m_color, linewidth=2, label="Median YoY")
            title_band = "(Median ± MAD)"

        apply_ib_style(ax, title=f"Price Distribution Band: {market.capitalize()} {title_band}", ylabel="YoY (%)")
        
        # 标注最新中位数点
        annotate_latest(ax, g, mid_col if lo_col else "median_pct", "Latest Median")
        
        plt.tight_layout()
        plt.savefig(out_dir / f"yoy_band_{market}.png", bbox_inches="tight")
        plt.close()

        # =========================
        # 2) Tail / breadth shares
        # =========================
        # Your metrics have: tail_down, share_95_100, breadth_up (>=100)
        # We draw what exists.
        plt.figure()

        any_line = False
        if "tail_down" in g.columns:
            plt.plot(g["date"], g["tail_down"], label=f"<= {g.get('low_tail_threshold', pd.Series([95])).iloc[0]:g}")
            any_line = True
        if "share_95_100" in g.columns:
            plt.plot(g["date"], g["share_95_100"], label="95-100")
            any_line = True
        if "tail_up" in g.columns:
            plt.plot(g["date"], g["tail_up"], label=">=100")
            any_line = True
        elif "breadth_up" in g.columns:
            plt.plot(g["date"], g["breadth_up"], label=">=100")
            any_line = True

        if any_line:
            plt.title(f"Tail / breadth shares — {market}")
            plt.xlabel("Date")
            plt.ylabel("Share of cities")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"tail_shares_{market}.png", dpi=200)
            plt.close()
        else:
            logging.warning(f"[Task D2] Skip tail shares for {market}: no tail/breadth columns.")
            plt.close()

        # =========================
        # 3) Dispersion
        # =========================
        # Your metrics definitely have mad_pct; iqr_pct may not exist.
        if "mad_pct" not in g.columns:
            logging.warning(f"[Task D2] Skip dispersion for {market}: missing mad_pct.")
            continue

        plt.figure()
        plt.plot(g["date"], g["mad_pct"], label="MAD")

        if "iqr_pct" in g.columns:
            plt.plot(g["date"], g["iqr_pct"], label="IQR")

        plt.title(f"Dispersion (compression proxy) — {market}")
        plt.xlabel("Date")
        plt.ylabel("Dispersion (pct points)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"dispersion_{market}.png", dpi=200)
        plt.close()

# =========================================================
# D3 / Report figures (for report_property.md)
# =========================================================

def make_property_report_figures(
    regime_csv: Path,
    leadlag_csv: Path,
    out_dir: Path = Path("output/property/figures"),
):
    """
    Report-facing figures:
    - regime_stabilization_prob.png
    - validation_leadlag_corr.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Regime stabilization probability
    # -------------------------
    if regime_csv.exists():
        # --- [补齐缺失的数据加载逻辑] ---
        reg = pd.read_csv(regime_csv)
        reg["date"] = pd.to_datetime(reg["date"], errors="coerce")
        reg = reg.dropna(subset=["date"])
        
        # 确定分组列名
        ht_col = "house_type" if "house_type" in reg.columns else (
            "market" if "market" in reg.columns else "ALL"
        )
        if ht_col == "ALL": reg["ALL"] = "ALL"
        # ----------------------------

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        for ht, g in reg.groupby(ht_col):
            g = g.sort_values("date")
            # 按照新房(蓝)/二手房(金)配色
            ax.plot(
                g["date"],
                pd.to_numeric(g["stabilization_prob"], errors="coerce"),
                label=str(ht),
                color=MARKET_COLORS.get(str(ht), "#6A737B"),
                linewidth=2.5
            )

        apply_ib_style(ax, title="Property Stabilization Probability (Monthly)", ylabel="Probability (0-1)")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, loc="upper left", **FONT_SETTINGS["legend"])
        
        plt.tight_layout()
        plt.savefig(out_dir / "regime_stabilization_prob.png", bbox_inches="tight")
        plt.close()
    else:
        logging.warning(f"[report_figures] missing regime csv: {regime_csv}")

    # -------------------------
    # 2) Lead / Lag correlation
    # -------------------------
    if leadlag_csv.exists():
        # --- [补齐缺失的数据加载逻辑] ---
        ll = pd.read_csv(leadlag_csv)
        # ----------------------------

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        # 使用投行标准循环：深夜蓝/高盛金
        cycle_colors = ["#00204E", "#A68B5B", "#6A737B"]
        
        for i, (s, g) in enumerate(ll.groupby("series")):
            g = g.sort_values("lag_months")
            ax.plot(
                pd.to_numeric(g["lag_months"], errors="coerce"),
                pd.to_numeric(g["corr"], errors="coerce"),
                marker="o",
                markersize=6,
                label=str(s),
                color=cycle_colors[i % len(cycle_colors)],
                linewidth=2
            )

        apply_ib_style(ax, title="Cross-Correlation: Signal vs Sales Activity", ylabel="Correlation", has_negative=True)
        ax.set_xlabel("Lag (months) | +lag = Sales Later (Signal Leads)", **FONT_SETTINGS["label"])
        ax.legend(frameon=False, loc="upper right", **FONT_SETTINGS["legend"])
        
        plt.tight_layout()
        plt.savefig(out_dir / "validation_leadlag_corr.png", bbox_inches="tight")
        plt.close()
    else:
        logging.warning(f"[report_figures] missing leadlag csv: {leadlag_csv}")

    apply_ib_style(ax, title="Cross-Correlation: Signal vs Sales Activity", ylabel="Correlation", has_negative=True)
    ax.set_xlabel("Lag (months) | +lag = Sales Later (Signal Leads)", **FONT_SETTINGS["label"])
    ax.legend(frameon=False, loc="upper right", **FONT_SETTINGS["legend"])
    
    plt.tight_layout()
    plt.savefig(out_dir / "validation_leadlag_corr.png", bbox_inches="tight")
    plt.close()
