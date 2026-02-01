from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Institutional Style Constants ---
COLORS = {
    "primary": "#00204E",     # Midnight Blue
    "secondary": "#6A737B",   # Slate Grey
    "highlight": "#A68B5B",   # Goldman Gold
    "risk": "#9E1B32",        # Alert Red
    "grid": "#D1D5D8"
}

FONT_SETTINGS = {
    "title": {"fontsize": 14, "fontweight": "bold", "family": "sans-serif"},
    "label": {"fontsize": 10, "family": "sans-serif"},
    "legend": {"fontsize": 9},
    "tick": {"labelsize": 9}
}

@dataclass(frozen=True)
class PlotPaths:
    features_dir: Path
    figures_dir: Path

# --- Robust Utility Functions ---

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    # Using utf-8-sig to handle potential BOM from Excel-exported CSVs
    return pd.read_csv(path, encoding="utf-8-sig")

def _parse_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Robust date parsing with cleaning for buy-side time-series."""
    if col not in df.columns:
        return df
    out = df.copy()
    # Coerce errors to NaT, then drop to prevent matplotlib plotting errors
    out[col] = pd.to_datetime(out[col], errors="coerce")
    original_count = len(out)
    out = out.dropna(subset=[col])
    
    if len(out) < original_count:
        logging.warning(f"Dropped {original_count - len(out)} rows with invalid dates.")
        
    return out.sort_values(col)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _pick_fiscal_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = set(df.columns)
    candidates = [
        ("tsf_lead_z", "target_z"),
        ("tsf_z", "target_z"),
        ("tsf_lead", "target"),
        ("tsf", "target"),
    ]
    for a, b in candidates:
        if a in cols and b in cols:
            return a, b
    
    # Fuzzy fallback
    tsf_like = [c for c in df.columns if "tsf" in c.lower()]
    tgt_like = [c for c in df.columns if "target" in c.lower() or "industrial" in c.lower()]
    if tsf_like and tgt_like:
        return tsf_like[0], tgt_like[0]

    raise ValueError("Cannot infer fiscal columns from: " + ", ".join(list(df.columns)))

# --- Buy-Side Styling Engine ---

def apply_ib_style(ax: plt.Axes, title: str, ylabel: str, has_negative: bool = False):
    """Applies institutional research standards to the plot."""
    ax.set_title(title, loc='left', **FONT_SETTINGS["title"], pad=20)
    ax.set_ylabel(ylabel, **FONT_SETTINGS["label"])
    
    # Grid & Spines
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', color=COLORS["secondary"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    
    # Ticks & Date Formatting
    ax.tick_params(axis='both', which='major', **FONT_SETTINGS["tick"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    if has_negative:
        ax.axhline(0, color=COLORS["primary"], linewidth=0.8, zorder=1)

def annotate_latest(ax: plt.Axes, df: pd.DataFrame, y_col: str, label: str = "Latest"):
    """Adds a callout for the most recent data point."""
    if df.empty: return
    latest = df.iloc[-1]
    ax.annotate(
        f"{label}: {latest[y_col]:.2f}",
        xy=(latest["date"], latest[y_col]),
        xytext=(15, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["highlight"], alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=COLORS["highlight"]),
        fontsize=8, fontweight='bold'
    )

# --- Main Plotting Tasks ---

def plot_fiscal_fuse(paths: PlotPaths) -> Path:
    fpath = paths.features_dir / "feature_fiscal_fuse.csv"
    df = _parse_date_col(_read_csv(fpath))
    xcol, ycol = _pick_fiscal_columns(df)
    d = df[["date", xcol, ycol]].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d["date"], d[xcol], color=COLORS["highlight"], label=f"Credit Lead ({xcol})", linewidth=2)
    ax.plot(d["date"], d[ycol], color=COLORS["primary"], label=f"Target ({ycol})", linewidth=2)
    
    apply_ib_style(ax, "Fiscal Fuse: Credit Cycles vs Real Activity", "Standardized Value (z-score)", has_negative=True)
    annotate_latest(ax, d, ycol)
    
    ax.legend(loc="upper left", frameon=False, **FONT_SETTINGS["legend"])
    
    _ensure_dir(paths.figures_dir)
    outpath = paths.figures_dir / "fiscal_fuse.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath

def plot_leadlag_corr(paths: PlotPaths) -> Path:
    fpath = paths.features_dir / "diagnostic_fiscal_leadlag_corr.csv"
    df = _read_csv(fpath)
    
    lag_col = next((c for c in df.columns if "lag" in c.lower()), df.columns[0])
    corr_col = next((c for c in df.columns if "corr" in c.lower()), df.columns[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[lag_col], df[corr_col], color=COLORS["secondary"], alpha=0.6, width=0.8)
    
    # Highlight peak correlation
    peak_idx = df[corr_col].abs().idxmax()
    ax.patches[peak_idx].set_color(COLORS["highlight"])
    ax.patches[peak_idx].set_alpha(1.0)

    ax.set_title("Cross-Correlation: Credit Leading Real Activity", loc='left', **FONT_SETTINGS["title"])
    ax.set_xlabel("Lag (Months) [Positive = Credit Leads]", **FONT_SETTINGS["label"])
    ax.set_ylabel("Correlation Coefficient", **FONT_SETTINGS["label"])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')

    _ensure_dir(paths.figures_dir)
    outpath = paths.figures_dir / "fiscal_leadlag_corr.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath

def plot_paradigm_shift(paths: PlotPaths) -> Path:
    fpath = paths.features_dir / "feature_paradigm_shift.csv"
    df = _parse_date_col(_read_csv(fpath))
    d = df.dropna(subset=["paradigm_index"])

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Old Economy in Risk Red, New Engine in Goldman Gold
    ax.plot(d["date"], d["new_comp"], color=COLORS["highlight"], label="New Engine (Composite)", linewidth=1.5)
    ax.plot(d["date"], d["old_comp"], color=COLORS["risk"], label="Old Economy (Composite)", linewidth=1.5, linestyle='--')
    ax.fill_between(d["date"], d["paradigm_index"], color=COLORS["primary"], alpha=0.1, label="Structural Delta")
    ax.plot(d["date"], d["paradigm_index"], color=COLORS["primary"], label="Paradigm Index (Net)", linewidth=2.5)
    
    apply_ib_style(ax, "Paradigm Shift Index: Structural Transition", "Scaled Intensity (pp)", has_negative=True)
    annotate_latest(ax, d, "paradigm_index", "Latest Shift")
    
    ax.legend(loc="upper left", frameon=False, ncol=2, **FONT_SETTINGS["legend"])
    
    _ensure_dir(paths.figures_dir)
    outpath = paths.figures_dir / "paradigm_shift.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath

def run_all_plots(paths: PlotPaths) -> None:
    logging.info("Starting professional plot generation...")
    try:
        plot_fiscal_fuse(paths)
        plot_leadlag_corr(paths)
        plot_paradigm_shift(paths)
        logging.info(f"Successfully wrote all plots to {paths.figures_dir}")
    except Exception as e:
        logging.error(f"Plotting failed: {str(e)}")
        raise