from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioSpec:
    # Inputs
    demand_mix_panel_csv: Path                 # output/domestic/demand_mix_panel.csv
    trade_panel_csv: Path                      # output/external/trade_two_point_panel_11月.csv (or ytd)
    property_regime_csv: Path | None = None    # output/property/monthly_regime_table.csv (optional)

    # Output
    output_dir: Path = Path("output/model")
    figures_dir: Path = Path("output/model/figures")

    # Baseline growth (e.g., a neutral prior)
    baseline_growth: float = 0.045  # 4.5% as a neutral anchor; you can change in config

    # Mapping (elasticities / sensitivities)
    beta_domestic: float = 0.0030   # growth change per 1 z of demand_mix_index
    beta_property: float = 0.0020   # growth change per 0.10 increase in stabilization prob
    beta_external: float = 0.0015   # growth change per 1% export yoy (approx proxy)

    # Monte Carlo controls
    n_sims: int = 2000
    seed: int = 42

    # Parameter uncertainty (optional): add realism without pretending we estimated a regression
    param_sigma_scale: float = 0.25  # 25% std around each beta


def _zscore_last(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((s.iloc[-1] - mu) / sd)


def _load_domestic_last_z(panel_path: Path) -> float:
    df = pd.read_csv(panel_path, parse_dates=["date"])
    if "demand_mix_index" not in df.columns:
        raise ValueError("demand_mix_panel.csv must contain demand_mix_index")
    df = df.sort_values("date")
    return _zscore_last(df["demand_mix_index"])


def _load_export_yoy_proxy(trade_panel_path: Path) -> float:
    """
    Use total exports % change between 2024 and 2025 for the chosen period as a crude proxy.
    We only have two points; this is NOT a time-series estimate.
    """
    df = pd.read_csv(trade_panel_path)
    if "region" not in df.columns:
        raise ValueError("trade panel must contain region")
    row = df[df["region"] == "总值"]
    if row.empty:
        # fallback: sum across rows
        v2024 = df["value_2024"].sum(skipna=True)
        v2025 = df["value_2025"].sum(skipna=True)
    else:
        v2024 = float(row["value_2024"].iloc[0])
        v2025 = float(row["value_2025"].iloc[0])

    if v2024 == 0 or not np.isfinite(v2024) or not np.isfinite(v2025):
        return float("nan")
    return float((v2025 - v2024) / v2024)  # e.g., 0.067


def _load_property_stabil_prob(property_regime_csv: Path | None) -> float:
    """
    Expect a table with date + a stabilization probability column.
    If absent, return NaN and model will ignore this channel.
    """
    if property_regime_csv is None or (not Path(property_regime_csv).exists()):
        return float("nan")

    df = pd.read_csv(property_regime_csv)
    # try common column names
    candidates = [c for c in df.columns if c.lower() in {"stabilization_prob", "stabil_prob", "regime_prob", "bottoming_prob"}]
    if not candidates:
        # soft match
        for c in df.columns:
            if "prob" in c.lower():
                candidates.append(c)
                break
    if not candidates:
        return float("nan")

    prob_col = candidates[0]
    # take latest non-null
    s = pd.to_numeric(df[prob_col], errors="coerce").dropna()
    if s.empty:
        return float("nan")
    p_last = float(s.iloc[-1])
    return p_last


def run_scenario_engine(spec: ScenarioSpec) -> Dict[str, Any]:
    logging.info("[Task G] Running scenario accounting engine (limited-data version)...")

    rng = np.random.default_rng(spec.seed)

    z_dom = _load_domestic_last_z(spec.demand_mix_panel_csv)
    ex_yoy = _load_export_yoy_proxy(spec.trade_panel_csv)
    p_stab = _load_property_stabil_prob(spec.property_regime_csv)

    # Convert inputs into growth deltas (deterministic center)
    # Domestic: per 1 z-score
    d_dom = spec.beta_domestic * z_dom if np.isfinite(z_dom) else 0.0

    # Property: interpret as +0.10 prob => +beta_property
    # So delta = beta_property * (p_stab - 0.5) / 0.10, anchored at 0.5 as neutral
    if np.isfinite(p_stab):
        d_prop = spec.beta_property * ((p_stab - 0.5) / 0.10)
    else:
        d_prop = 0.0

    # External: per 1% export growth (0.01). ex_yoy is in fraction.
    d_ext = spec.beta_external * (ex_yoy / 0.01) if np.isfinite(ex_yoy) else 0.0

    center = spec.baseline_growth + d_dom + d_prop + d_ext

    # Monte Carlo: introduce parameter uncertainty (NOT estimation)
    beta_dom_sims = rng.normal(spec.beta_domestic, abs(spec.beta_domestic) * spec.param_sigma_scale, size=spec.n_sims)
    beta_prop_sims = rng.normal(spec.beta_property, abs(spec.beta_property) * spec.param_sigma_scale, size=spec.n_sims)
    beta_ext_sims = rng.normal(spec.beta_external, abs(spec.beta_external) * spec.param_sigma_scale, size=spec.n_sims)

    dom_term = beta_dom_sims * (z_dom if np.isfinite(z_dom) else 0.0)
    prop_term = beta_prop_sims * (((p_stab - 0.5) / 0.10) if np.isfinite(p_stab) else 0.0)
    ext_term = beta_ext_sims * ((ex_yoy / 0.01) if np.isfinite(ex_yoy) else 0.0)

    growth = spec.baseline_growth + dom_term + prop_term + ext_term

    # Save outputs
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    dist_df = pd.DataFrame({"growth": growth})
    dist_path = spec.output_dir / "growth_distribution.csv"
    dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")

    # Sensitivity table (one-at-a-time shocks, deterministic)
    sens = []
    sens.append({"factor": "domestic_z", "value": z_dom, "beta": spec.beta_domestic, "contrib": d_dom})
    sens.append({"factor": "property_stabil_prob", "value": p_stab, "beta": spec.beta_property, "contrib": d_prop})
    sens.append({"factor": "export_yoy_proxy", "value": ex_yoy, "beta": spec.beta_external, "contrib": d_ext})

    sens_df = pd.DataFrame(sens)
    sens_path = spec.output_dir / "sensitivity_table.csv"
    sens_df.to_csv(sens_path, index=False, encoding="utf-8-sig")

        # -----------------------------
    # Scenario grid (deterministic)
    # -----------------------------
    # Define shocks in "interpretable units":
    # - domestic_z: +/- 0.5 z
    # - property_prob: +/- 0.10
    # - export_yoy: +/- 0.03 (3ppt)
    grid = []

    domestic_shocks = {
        "domestic_soft": -0.5,
        "domestic_base": 0.0,
        "domestic_strong": +0.5,
    }
    property_shocks = {
        "property_downside": -0.10,
        "property_base": 0.0,
        "property_upside": +0.10,
    }
    external_shocks = {
        "external_weak": -0.03,
        "external_base": 0.0,
        "external_strong": +0.03,
    }

    # Center on latest readings, then apply shocks
    z0 = z_dom if np.isfinite(z_dom) else 0.0
    p0 = p_stab if np.isfinite(p_stab) else 0.5  # neutral anchor
    e0 = ex_yoy if np.isfinite(ex_yoy) else 0.0

    for d_name, dz in domestic_shocks.items():
        for p_name, dp in property_shocks.items():
            for e_name, de in external_shocks.items():
                z = z0 + dz
                p = p0 + dp
                e = e0 + de

                contrib_dom = spec.beta_domestic * z
                contrib_prop = spec.beta_property * ((p - 0.5) / 0.10)
                contrib_ext = spec.beta_external * ((e) / 0.01)

                g = spec.baseline_growth + contrib_dom + contrib_prop + contrib_ext

                grid.append(
                    {
                        "scenario": f"{d_name}|{p_name}|{e_name}",
                        "domestic_z": z,
                        "property_stabil_prob": p,
                        "export_yoy_proxy": e,
                        "contrib_domestic": contrib_dom,
                        "contrib_property": contrib_prop,
                        "contrib_external": contrib_ext,
                        "growth": g,
                    }
                )

    grid_df = pd.DataFrame(grid).sort_values("growth", ascending=False)
    grid_path = spec.output_dir / "scenario_grid.csv"
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    logging.info(f"[Task G] Wrote: {grid_path}")

    return {
        "inputs": {"z_domestic": z_dom, "export_yoy_proxy": ex_yoy, "property_stabil_prob": p_stab},
        "center_growth": center,
        "dist_path": dist_path,
        "sens_path": sens_path,
        "grid_path": grid_path,

    }
