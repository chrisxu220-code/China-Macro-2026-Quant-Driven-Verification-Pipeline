from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CAProxySpec:
    # reuse trade outputs
    trade_tidy_csv: Path
    output_dir: Path = Path("output/external")
    # which period to report
    period: str = "1至11月"  # or "11月"


def run_ca_proxy(spec: CAProxySpec) -> Path:
    """
    Minimal, honest CA proxy:
    - We only have goods trade table here.
    - CA ≠ trade balance, but goods balance is a big component.
    So we produce a "goods balance proxy" and explicitly label it as such.
    """
    logging.info("[Task E] Running CA proxy (goods balance proxy)...")

    df = pd.read_csv(spec.trade_tidy_csv)
    df = df[df["period"] == spec.period].copy()

    # We only have export values in the parsed trade_tidy from trade.py.
    # If later you add import columns, we can extend to (exports - imports).
    # For now, we create a "CA proxy readiness" output that documents limitation.
    # We'll still compute total exports for 2024/2025.
    total = (
        df[df["region"] == "总值"]
        .pivot_table(index="period", columns="year", values="export_value", aggfunc="sum")
        .reset_index()
    )

    v2024 = float(total.get(2024, pd.Series([np.nan])).iloc[0]) if 2024 in total.columns else np.nan
    v2025 = float(total.get(2025, pd.Series([np.nan])).iloc[0]) if 2025 in total.columns else np.nan

    out = pd.DataFrame(
        [
            {
                "period": spec.period,
                "metric": "goods_exports_total_rmb_10k",
                "value_2024": v2024,
                "value_2025": v2025,
                "delta": v2025 - v2024 if np.isfinite(v2024) and np.isfinite(v2025) else np.nan,
                "pct_change": (v2025 - v2024) / v2024 if np.isfinite(v2024) and v2024 != 0 else np.nan,
                "notes": "This is exports only (from sheet '出口'). Not a full current account; imports/services/income not included.",
            }
        ]
    )

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = spec.output_dir / f"ca_proxy_{spec.period}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info(f"[Task E] Wrote: {out_path}")
    logging.info("[Task E] CA proxy finished ✅")
    return out_path
