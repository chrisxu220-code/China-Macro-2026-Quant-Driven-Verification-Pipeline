from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config (edit if you want)
# -----------------------------
REPO_ROOT = Path(".")
PROPERTY_DIR = REPO_ROOT / "output" / "property"

REGIME_CSV = PROPERTY_DIR / "monthly_regime_table.csv"
METRICS_CSV = PROPERTY_DIR / "price_distribution_metrics.csv"
LEADLAG_CSV = PROPERTY_DIR / "validation_leadlag_corr.csv"

DEFAULT_NEUTRAL = 100.0
DEFAULT_LOW_TAIL = 95.0


# -----------------------------
# Helpers
# -----------------------------
def _fmt_pct(x: Any, digits: int = 1) -> str:
    """Format share (0-1) as percent string like '7.7%'."""
    try:
        v = float(x)
    except Exception:
        return "NA"
    if np.isnan(v):
        return "NA"
    return f"{v*100:.{digits}f}%"


def _fmt_num(x: Any, digits: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if np.isnan(v):
        return "NA"
    return f"{v:.{digits}f}"


def _parse_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.dropna(subset=[col])
    return df


def _month_str(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m")


def _pick_house_type_focus(reg: pd.DataFrame) -> str:
    # If the regime table has house_type, use ALL / 新房 / 二手房 logic.
    if "house_type" in reg.columns:
        u = [str(x) for x in reg["house_type"].dropna().unique().tolist()]
        if "ALL" in u:
            return "ALL (pooled new+existing)"
        if "新房" in u and "二手房" in u:
            return "New vs Existing"
        if len(u) == 1:
            return u[0]
        return u[0]
    # otherwise fallback
    if "market" in reg.columns:
        u = [str(x) for x in reg["market"].dropna().unique().tolist()]
        if len(u) == 1:
            return u[0]
        if u:
            return " / ".join(u[:2])
    return "ALL"


def _find_turning_window(reg: pd.DataFrame) -> str:
    """
    Heuristic: first month where stabilization_prob crosses above 0.50 (and stays above
    for >=2 months if possible). Fallback: window around max.
    """
    g = reg.sort_values("date").copy()
    p = pd.to_numeric(g["stabilization_prob"], errors="coerce")
    g = g.assign(p=p).dropna(subset=["p"])
    if g.empty:
        return "NA"

    above = g["p"] >= 0.5
    # Find first index of >=0.5
    idxs = np.where(above.values)[0]
    if len(idxs) == 0:
        # fallback: around peak
        i = int(np.nanargmax(g["p"].values))
        start = max(0, i - 1)
        end = min(len(g) - 1, i + 1)
        return f"{_month_str(g.iloc[start]['date'])}–{_month_str(g.iloc[end]['date'])}"

    i0 = int(idxs[0])
    # Try to make it a window of 2-3 months starting at i0
    start = i0
    end = min(len(g) - 1, i0 + 2)
    return f"{_month_str(g.iloc[start]['date'])}–{_month_str(g.iloc[end]['date'])}"


def _latest_row(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("date")
    return df.iloc[-1]


def _metric_latest(metrics: pd.DataFrame, market: str) -> Optional[pd.Series]:
    g = metrics[metrics["market"].astype(str) == market].copy()
    if g.empty:
        return None
    g = g.sort_values("date")
    return g.iloc[-1]


def _best_leadlag(ll: pd.DataFrame) -> Tuple[str, str]:
    """
    Return (BEST_LAG, PEAK_CORR).
    Only consider positive lag as 'signal leads sales'.
    If nothing usable, return ("NA","NA").
    """
    if ll.empty:
        return "NA", "NA"
    ll = ll.copy()
    ll["lag_months"] = pd.to_numeric(ll["lag_months"], errors="coerce")
    ll["corr"] = pd.to_numeric(ll["corr"], errors="coerce")
    ll = ll.dropna(subset=["lag_months", "corr"])
    if ll.empty:
        return "NA", "NA"

    # focus on +lag (sales later)
    pos = ll[ll["lag_months"] > 0].copy()
    if pos.empty:
        # fallback: pick max abs corr overall
        j = int(np.nanargmax(np.abs(ll["corr"].values)))
        best_lag = int(ll.iloc[j]["lag_months"])
        peak_corr = float(ll.iloc[j]["corr"])
        return str(best_lag), _fmt_num(peak_corr, 2)

    # pick max correlation among positive lags
    j = int(np.nanargmax(pos["corr"].values))
    best_lag = int(pos.iloc[j]["lag_months"])
    peak_corr = float(pos.iloc[j]["corr"])
    return str(best_lag), _fmt_num(peak_corr, 2)


def _replace_placeholders(md_text: str, mapping: Dict[str, str]) -> str:
    """
    Replace {{KEY}} with mapping[KEY], leave as-is if missing.
    """
    def repl(m: re.Match) -> str:
        k = m.group(1).strip()
        return mapping.get(k, m.group(0))
    return re.sub(r"\{\{([^}]+)\}\}", repl, md_text)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # ---- load csvs
    if not REGIME_CSV.exists():
        raise FileNotFoundError(f"Missing: {REGIME_CSV}")
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing: {METRICS_CSV}")

    reg = pd.read_csv(REGIME_CSV)
    reg = _parse_date_col(reg, "date")
    if "stabilization_prob" not in reg.columns:
        raise KeyError(f"monthly_regime_table.csv missing stabilization_prob. cols={list(reg.columns)}")

    # If multiple house_types exist, use ALL if present else first
    if "house_type" in reg.columns:
        u = [str(x) for x in reg["house_type"].dropna().unique().tolist()]
        focus_ht = "ALL" if "ALL" in u else u[0]
        reg_focus = reg[reg["house_type"].astype(str) == focus_ht].copy()
    else:
        focus_ht = "ALL"
        reg_focus = reg.copy()

    reg_focus = reg_focus.sort_values("date")
    latest_reg = _latest_row(reg_focus)
    end_month = _month_str(latest_reg["date"])
    start_month = _month_str(reg_focus["date"].min())

    latest_prob = float(pd.to_numeric(latest_reg["stabilization_prob"], errors="coerce"))
    # city coverage: try columns in priority order
    num_cities = None
    for c in ["num_cities", "n_cities", "city_count", "n"]:
        if c in reg_focus.columns:
            num_cities = int(pd.to_numeric(latest_reg[c], errors="coerce"))
            break
    if num_cities is None:
        num_cities = "NA"

    turning_window = _find_turning_window(reg_focus)
    house_type_focus = _pick_house_type_focus(reg)

    # ---- metrics
    metrics = pd.read_csv(METRICS_CSV)
    metrics = _parse_date_col(metrics, "date")

    # thresholds from file if present
    neutral = DEFAULT_NEUTRAL
    low_tail = DEFAULT_LOW_TAIL
    if "neutral_threshold" in metrics.columns:
        neutral = float(pd.to_numeric(metrics["neutral_threshold"].dropna().iloc[0], errors="coerce"))
    if "low_tail_threshold" in metrics.columns:
        low_tail = float(pd.to_numeric(metrics["low_tail_threshold"].dropna().iloc[0], errors="coerce"))

    latest_new = _metric_latest(metrics, "new")
    latest_existing = _metric_latest(metrics, "existing")

    # Fill market metrics safely
    def pull_market(row: Optional[pd.Series]) -> Dict[str, str]:
        if row is None:
            return {
                "BREADTH_UP": "NA",
                "TAIL_DOWN": "NA",
                "SHARE_95_100": "NA",
                "MAD_PCT": "NA",
                "MEDIAN_PCT": "NA",
                "NUM_CITIES": "NA",
            }
        out = {}
        out["BREADTH_UP"] = _fmt_pct(row.get("breadth_up", np.nan))
        out["TAIL_DOWN"] = _fmt_pct(row.get("tail_down", np.nan))
        out["SHARE_95_100"] = _fmt_pct(row.get("share_95_100", np.nan))
        # mad_pct/median_pct are in pct-pts already in your file (looks like 0.01 = 1ppt)
        out["MAD_PCT"] = _fmt_num(row.get("mad_pct", np.nan), 3)
        out["MEDIAN_PCT"] = _fmt_num(row.get("median_pct", np.nan), 3)
        out["NUM_CITIES"] = str(int(pd.to_numeric(row.get("num_cities", np.nan), errors="coerce"))) if "num_cities" in row else "NA"
        return out

    new_m = pull_market(latest_new)
    ex_m = pull_market(latest_existing)

    # ---- lead/lag
    best_lag = "NA"
    peak_corr = "NA"
    if LEADLAG_CSV.exists():
        ll = pd.read_csv(LEADLAG_CSV)
        best_lag, peak_corr = _best_leadlag(ll)

    # ---- placeholder mapping
    run_date = datetime.now().strftime("%Y-%m-%d")
    mapping: Dict[str, str] = {
        "RUN_DATE": run_date,
        "START_MONTH": start_month,
        "END_MONTH": end_month,
        "TURNING_WINDOW": turning_window,
        "LATEST_PROB": _fmt_num(latest_prob, 2),
        "HOUSE_TYPE_FOCUS": house_type_focus,
        "NEUTRAL_THRESHOLD": _fmt_num(neutral, 0),
        "LOW_TAIL_THRESHOLD": _fmt_num(low_tail, 0),
        "NUM_CITIES": str(num_cities),
        "BEST_LAG": best_lag,
        "PEAK_CORR": peak_corr,
        # New
        "BREADTH_UP_NEW": new_m["BREADTH_UP"],
        "TAIL_DOWN_NEW": new_m["TAIL_DOWN"],
        "SHARE_95_100_NEW": new_m["SHARE_95_100"],
        "MAD_PCT_NEW": new_m["MAD_PCT"],
        "MEDIAN_PCT_NEW": new_m["MEDIAN_PCT"],
        "NUM_CITIES_NEW": new_m["NUM_CITIES"],
        # Existing
        "BREADTH_UP_EXISTING": ex_m["BREADTH_UP"],
        "TAIL_DOWN_EXISTING": ex_m["TAIL_DOWN"],
        "SHARE_95_100_EXISTING": ex_m["SHARE_95_100"],
        "MAD_PCT_EXISTING": ex_m["MAD_PCT"],
        "MEDIAN_PCT_EXISTING": ex_m["MEDIAN_PCT"],
        "NUM_CITIES_EXISTING": ex_m["NUM_CITIES"],
    }

    # ---- write placeholders json
    out_json = PROPERTY_DIR / "report_placeholders.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote: {out_json}")

    # ---- optional: fill a markdown template
    # By default, try property_report.md if exists in repo root
    template_candidates = [
        REPO_ROOT / "property_report.md",
        REPO_ROOT / "property_report_rewrite.md",
    ]
    template = next((p for p in template_candidates if p.exists()), None)
    if template:
        md_text = template.read_text(encoding="utf-8")
        filled = _replace_placeholders(md_text, mapping)
        out_md = PROPERTY_DIR / "property_report_filled.md"
        out_md.write_text(filled, encoding="utf-8")
        print(f"[ok] wrote: {out_md}  (from template {template.name})")
    else:
        print("[info] No template found (property_report.md / property_report_rewrite.md). Skipped filling MD.")


if __name__ == "__main__":
    main()
