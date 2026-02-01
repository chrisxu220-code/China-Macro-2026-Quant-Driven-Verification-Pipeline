from __future__ import annotations
from .data_registry import run_data_ingestion
from .features.build import FeatureBuildConfig, run_feature_build
from .features.fiscal import FiscalFuseSpec
from .features.structural_shift import ParadigmShiftSpec
from .property.regime_signal import RegimeSpec, run_regime_signal
from .property.validation import ValidationSpec, run_sales_inventory_validation
from .external.trade import TradeSpec, run_trade
from .external.ca_proxy import CAProxySpec, run_ca_proxy
from .report.build import ReportSpec, build_main_report, write_readme_pitch
from .property.housing_clean import CleanSpec, build_panel
from .property.heatmap_dashboard import HeatmapSpec, make_dashboard
from .model.nowcast import ModelSpec, AnchorLine, run_model


import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

from .config import AppConfig, load_config


def setup_logging(cfg: AppConfig) -> None:
    level = getattr(logging, cfg.log_level, logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if cfg.log_to_file:
        cfg.paths.ensure()
        log_path = cfg.paths.logs_dir / cfg.log_filename
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def step_smoke_test(cfg: AppConfig) -> None:
    logging.info("Running smoke_test...")
    logging.info(f"Project: {cfg.project_name}")
    logging.info(f"Repo root: {cfg.paths.repo_root}")
    logging.info(f"Data dir: {cfg.paths.data_dir}")
    logging.info(f"Output dir: {cfg.paths.output_dir}")
    logging.info(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    if pd is None:
        logging.warning("pandas not installed/importable yet (ok for Task A).")
    else:
        logging.info(f"pandas version: {pd.__version__}")

def step_load_and_validate_data(cfg: AppConfig) -> None:
    import logging
    from pathlib import Path

    excel_path = Path(cfg.raw.get("inputs", {}).get("data_excel_path", ""))
    if not excel_path.exists():
        raise FileNotFoundError(
            f"inputs.data_excel_path not found: {excel_path}. "
            f"Please set it in config.yaml (Downloads path is ok)."
        )

    registry_path = cfg.paths.repo_root / "data" / "registry.yaml"
    run_data_ingestion(
        excel_path=excel_path,
        registry_path=registry_path,
        output_dir=cfg.paths.output_dir,
    )
    logging.info("[Task B] Data ingestion + QA finished ✅")

def step_build_features(cfg: AppConfig) -> None:
    import logging
    from pathlib import Path

    processed_long = cfg.paths.output_dir / "processed" / "timeseries_long.csv"
    if not processed_long.exists():
        raise FileNotFoundError(f"Missing processed long csv: {processed_long}. Run Task B first.")

    feat_out = cfg.paths.output_dir / "features"
    feat_out.mkdir(parents=True, exist_ok=True)

    fcfg_raw = cfg.raw.get("features", {})
    fcfg = FeatureBuildConfig(
        seasonality_method=str(fcfg_raw.get("seasonality_method", "merge_jan_feb")),
        dedup_method=str(fcfg_raw.get("dedup_method", "mean")),
    )

    # fiscal spec (optional)
    fiscal_spec = None
    f_raw = fcfg_raw.get("fiscal_fuse", {}) if isinstance(fcfg_raw, dict) else {}
    if isinstance(f_raw, dict) and f_raw.get("tsf_series") and f_raw.get("target_series"):
        fiscal_spec = FiscalFuseSpec(
            tsf_series=str(f_raw["tsf_series"]),
            target_series=str(f_raw["target_series"]),
            lead_months=int(f_raw.get("lead_months", 12)),
            zscore=bool(f_raw.get("zscore", True)),
        )

    # paradigm shift spec (optional)
    shift_spec = None
    p_raw = fcfg_raw.get("paradigm_shift", {}) if isinstance(fcfg_raw, dict) else {}
    if isinstance(p_raw, dict) and p_raw.get("new_engine_series") and p_raw.get("old_economy_series"):
        shift_spec = ParadigmShiftSpec(
            new_engine_series=[str(x) for x in p_raw.get("new_engine_series", [])],
            old_economy_series=[str(x) for x in p_raw.get("old_economy_series", [])],
            scale=str(p_raw.get("scale", "robust")),
            composite=str(p_raw.get("composite", "diff")),
            winsorize=float(p_raw.get("winsorize", 0.01)),
        )

    logging.info("[Task C] Starting feature build...")
    run_feature_build(
        processed_long_csv=processed_long,
        output_dir=feat_out,
        cfg=fcfg,
        fiscal_spec=fiscal_spec,
        shift_spec=shift_spec,
    )

def step_property_distribution(cfg: AppConfig) -> None:
    import logging
    from pathlib import Path

    # ---- D2: price distribution ----
    from src.property.price_distribution import (
        PriceWorkbookSpec,
        run_property_price_distribution,
    )

    # ---- D2: figures (from metrics csv) ----
    from src.property.figures import make_property_figures

    # ---- D2 extension: sales clean ----
    from src.property.sales import (
        SalesWorkbookSpec,
        run_property_sales_panel,
    )

    # ---- D3: regime + validation ----
    from src.property.regime_signal import RegimeSpec, run_regime_signal
    from src.property.validation import ValidationSpec, run_sales_inventory_validation

    # ---- D3: report figures (new, depends on regime + validation) ----
    from src.property.figures import make_property_report_figures

    # ---- read property config ----
    prop_cfg = cfg.raw.get("property", {})
    if not prop_cfg:
        raise ValueError("Missing 'property' section in config.yaml")

    price_path = Path(prop_cfg.get("price_xlsx_path", ""))
    sales_path = Path(prop_cfg.get("sales_xlsx_path", ""))

    if not price_path.exists():
        raise FileNotFoundError(f"Price workbook not found: {price_path}")

    output_dir = Path(prop_cfg.get("output_dir", "output/property"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dist_cfg = prop_cfg.get("distribution", {})
    figures_cfg = prop_cfg.get("figures", {})

    figs_out_dir = Path(figures_cfg.get("out_dir", output_dir / "figures"))
    figs_out_dir.mkdir(parents=True, exist_ok=True)

    low_tail = float(dist_cfg.get("low_tail_threshold", 95.0))
    neutral = float(dist_cfg.get("neutral_threshold", 100.0))
    dist_field = str(dist_cfg.get("distribution_field", "yoy_index"))

    # =========================================================
    # D2 (core): price distribution -> city panel + metrics
    # =========================================================
    spec = PriceWorkbookSpec(
        excel_path=price_path,
        output_dir=output_dir,
        low_tail_threshold=low_tail,
        neutral_threshold=neutral,
        distribution_field=dist_field,
    )

    logging.info("[Task D2] Running property price distribution...")
    panel_path, metrics_path = run_property_price_distribution(spec)
    panel_path = Path(panel_path)
    metrics_path = Path(metrics_path)
    logging.info(f"[Task D2] Wrote panel:   {panel_path}")
    logging.info(f"[Task D2] Wrote metrics: {metrics_path}")

    # =========================================================
    # D2 extension: sales workbook -> monthly sales panel CSV
    # =========================================================
    sales_csv = output_dir / "housing_sales_clean_monthly.csv"

    if sales_path and sales_path.exists():
        logging.info("[Task D2] Running property sales clean...")
        sales_spec = SalesWorkbookSpec(
            excel_path=sales_path,
            output_dir=output_dir,
        )
        sales_out = run_property_sales_panel(sales_spec)
        sales_csv = Path(sales_out)
        logging.info(f"[Task D2] Wrote sales monthly: {sales_csv}")
    else:
        logging.warning(f"[Task D2] Sales workbook not found, skip: {sales_path}")

    # =========================================================
    # D3: regime signal (uses city panel)
    # =========================================================
    logging.info("[Task D3] Running regime signal...")
    regime_out = run_regime_signal(RegimeSpec(
        price_city_panel_csv=panel_path,
        output_dir=output_dir,
        figures_dir=figs_out_dir,
        low_tail_threshold=low_tail,
        neutral_threshold=neutral,
        k_improve=3,
    ))
    regime_out = Path(regime_out)
    logging.info(f"[Task D3] Wrote regime table: {regime_out}")

    # =========================================================
    # D3: validation (uses sales monthly + regime table)
    # =========================================================
    if sales_csv.exists():
        logging.info("[Task D3] Running sales/inventory validation...")
        val_out = run_sales_inventory_validation(ValidationSpec(
            sales_monthly_csv=output_dir / "housing_sales_clean_monthly.csv",
            regime_table_csv=output_dir / "monthly_regime_table.csv",
            output_dir=output_dir,
            year=int(prop_cfg.get("year", 0)) or None,
        ))
        logging.info(f"[Task D3] Wrote validation table: {val_out}")
    else:
        logging.warning(f"[Task D3] Missing sales monthly csv, skip validation: {sales_csv}")

    # =========================================================
    # D2 figures (from metrics csv) — depends only on metrics
    # =========================================================
    if figures_cfg.get("enabled", False):
        logging.info("[Task D2] Generating property figures...")
        make_property_figures(
            metrics_csv=metrics_path,
            out_dir=figs_out_dir,
        )
        logging.info(f"[Task D2] Wrote figures to: {figs_out_dir}")

    # =========================================================
    # D3 report figures — MUST come after validation
    # =========================================================
    logging.info("[Task D3] Generating report-facing figures...")
    make_property_report_figures(
        regime_csv=output_dir / "monthly_regime_table.csv",
        leadlag_csv=output_dir / "validation_leadlag_corr.csv",
        out_dir=figs_out_dir,
    )
    logging.info(f"[Task D3] Wrote report figures to: {figs_out_dir}")

def _resolve_path(cfg: AppConfig, p: str) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (cfg.paths.repo_root / x)

def step_property_dashboard_heatmap(cfg: AppConfig) -> None:
    import logging
    import pandas as pd

    prop_cfg = cfg.raw.get("property", {})
    if not prop_cfg:
        raise ValueError("Missing 'property' section in config.yaml")

    price_xlsx = _resolve_path(cfg, str(prop_cfg.get("price_xlsx_path", "")))
    panel_csv  = _resolve_path(cfg, str(prop_cfg.get("panel_csv_path", "")))
    out_png    = _resolve_path(cfg, str(prop_cfg.get("heatmap_png_path", "")))

    start_date = str(prop_cfg.get("start_date", "2025-01-01"))
    row_gap    = float(prop_cfg.get("row_gap", 1.4))

    panel_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 1) clean -> panel csv
    spec_clean = CleanSpec(price_xlsx_path=price_xlsx, out_csv_path=panel_csv)
    df_panel = build_panel(spec_clean)
    df_panel.to_csv(panel_csv, index=False, encoding="utf-8-sig")
    logging.info(f"[Property Heatmap] panel -> {panel_csv}")

    # 2) heatmap -> png
    spec_hm = HeatmapSpec(panel_csv_path=panel_csv, out_png_path=out_png, start_date=start_date, row_gap=row_gap)
    df = pd.read_csv(panel_csv)
    df["date"] = pd.to_datetime(df["date"])
    make_dashboard(df, spec_hm)
    logging.info(f"[Property Heatmap] heatmap -> {out_png}")


#Step E
def step_external_trade(cfg: AppConfig) -> None:
    import logging
    from pathlib import Path

    ext_cfg = cfg.raw.get("external", {})
    if not ext_cfg:
        raise ValueError("Missing 'external' section in config.yaml")

    # Prefer inputs.data_excel_path (your standard raw excel entry)
    excel_path = Path(cfg.raw.get("inputs", {}).get("data_excel_path", ""))
    if not excel_path.exists():
        raise FileNotFoundError(f"inputs.data_excel_path not found: {excel_path}")

    output_dir = Path(ext_cfg.get("output_dir", "output/external"))
    figures_dir = Path(ext_cfg.get("figures_dir", "output/external/figures"))
    sheet = str(ext_cfg.get("sheet", "出口"))
    headline_period = str(ext_cfg.get("headline_period", "11月"))

    # Optional bucket map (DM/EM)
    bucket_map = ext_cfg.get("bucket_map", None)

    spec = TradeSpec(
        excel_path=excel_path,
        sheet=sheet,
        output_dir=output_dir,
        figures_dir=figures_dir,
        headline_period=headline_period,
        bucket_map=bucket_map if isinstance(bucket_map, dict) else None,
    )
    outputs = run_trade(spec)

    # CA proxy (goods exports proxy; honest limitation)
    ca_period = str(ext_cfg.get("ca_period", "1至11月"))
    ca_spec = CAProxySpec(
        trade_tidy_csv=outputs["trade_tidy"],
        output_dir=output_dir,
        period=ca_period,
    )
    run_ca_proxy(ca_spec)

    # Write a minimal report stub into report/
    report_dir = cfg.paths.repo_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report_external.md"
    if not report_path.exists():
        report_path.write_text(..., encoding="utf-8")
        logging.info(f"[Task E] Wrote report: {report_path}")
    else:
        logging.info(f"[Task E] Kept existing report: {report_path}")


#Step F
def step_domestic_demand(cfg: AppConfig) -> None:
    import logging
    from pathlib import Path
    import pandas as pd

    from src.domestic.consumption import ConsumptionSpec, run_consumption
    from src.domestic.investment import InvestmentSpec, run_investment
    from src.domestic.fiscal import FiscalSpec, run_fiscal

    processed_long = cfg.paths.output_dir / "processed" / "timeseries_long.csv"
    if not processed_long.exists():
        raise FileNotFoundError(f"Missing: {processed_long}. Run Task B first.")

    dom_cfg = cfg.raw.get("domestic", {})
    if not dom_cfg:
        raise ValueError("Missing 'domestic' section in config.yaml")

    out_dir = Path(dom_cfg.get("output_dir", "output/domestic"))
    fig_dir = Path(dom_cfg.get("figures_dir", "output/domestic/figures"))
    method = str(dom_cfg.get("method", "zscore_mean"))

    c_csv = run_consumption(
        ConsumptionSpec(processed_long_csv=processed_long, output_dir=out_dir, figures_dir=fig_dir,
                        series=list(dom_cfg.get("consumption_series", [])), method=method)
    )
    i_csv = run_investment(
        InvestmentSpec(processed_long_csv=processed_long, output_dir=out_dir, figures_dir=fig_dir,
                       series=list(dom_cfg.get("investment_series", [])), method=method)
    )
    f_csv = run_fiscal(
        FiscalSpec(processed_long_csv=processed_long, output_dir=out_dir, figures_dir=fig_dir,
                   series=list(dom_cfg.get("fiscal_series", [])), method=method)
    )

    # merge into demand mix panel
    c = pd.read_csv(c_csv, parse_dates=["date"])
    i = pd.read_csv(i_csv, parse_dates=["date"])
    f = pd.read_csv(f_csv, parse_dates=["date"])

    panel = c.merge(i, on="date", how="outer").merge(f, on="date", how="outer").sort_values("date")

    # simple overall index (equal weight; you can customize weights later)
    panel["demand_mix_index"] = panel[["consumption_index", "investment_index", "fiscal_index"]].mean(axis=1, skipna=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "demand_mix_panel.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    logging.info(f"[Task F] Wrote: {panel_path}")

    # write report stub
    report_dir = cfg.paths.repo_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rp = report_dir / "report_domestic.md"
    rp.write_text(
        "\n".join(
            [
                "# Domestic Demand Mix (Task F) — Proxy-based Dashboard",
                "",
                "## What this module can and cannot prove",
                "- **Can:** track verifiable demand proxies for consumption/investment/fiscal; summarize mix shifts as an index.",
                "- **Cannot:** claim national-accounts contribution decomposition unless expenditure-side GDP components are available.",
                "",
                "## Outputs",
                f"- `{panel_path}`",
                f"- `{fig_dir / 'consumption_index.png'}`",
                f"- `{fig_dir / 'investment_index.png'}`",
                f"- `{fig_dir / 'fiscal_index.png'}`",
                "",
                "## Interpretation (template)",
                "- Consumption proxy vs fiscal proxy: is policy offsetting private weakness?",
                "- Investment proxy: is there a modest rebound consistent with GS narrative?",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logging.info(f"[Task F] Wrote report: {rp}")

#Step G
def step_model_scenario(cfg: AppConfig) -> None:
    import logging
    from pathlib import Path

    # Inputs from previous tasks
    demand_mix = cfg.paths.repo_root / "output/domestic/demand_mix_panel.csv"
    trade_panel = cfg.paths.repo_root / "output/external/trade_two_point_panel_11月.csv"

    if not demand_mix.exists():
        raise FileNotFoundError(f"Missing: {demand_mix} (run domestic_demand first)")
    if not trade_panel.exists():
        raise FileNotFoundError(f"Missing: {trade_panel} (run external_trade first)")

    # property regime optional
    prop_regime = cfg.paths.repo_root / "output/property/monthly_regime_table.csv"
    prop_regime_path = prop_regime if prop_regime.exists() else None

    mcfg = cfg.raw.get("model", {})

    # --- add anchors (hard-coded or from config) ---
    anchors = [
        AnchorLine(label="IMF (latest)", value=0.045, linestyle=":", linewidth=1.6, alpha=0.75),
        AnchorLine(label="Goldman Sachs (latest)", value=0.048, linestyle="--", linewidth=1.6, alpha=0.75),
    ]

    spec = ModelSpec(
        demand_mix_panel_csv=demand_mix,
        trade_panel_csv=trade_panel,
        property_regime_csv=prop_regime_path,
        output_dir=Path(mcfg.get("output_dir", "output/model")),
        figures_dir=Path(mcfg.get("figures_dir", "output/model/figures")),
        baseline_growth=float(mcfg.get("baseline_growth", 0.045)),
        beta_domestic=float(mcfg.get("beta_domestic", 0.0030)),
        beta_property=float(mcfg.get("beta_property", 0.0020)),
        beta_external=float(mcfg.get("beta_external", 0.0015)),
        n_sims=int(mcfg.get("n_sims", 2000)),
        seed=int(cfg.seed),
        param_sigma_scale=float(mcfg.get("param_sigma_scale", 0.25)),
        anchors=anchors,  # ✅ NEW
    )


    out = run_model(spec)

    # write report stub
    report_dir = cfg.paths.repo_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rp = report_dir / "report_model.md"
    grid_path = out.get("grid_path")
    rp.write_text(
        "\n".join(
            [
                "# Model (Task G) — Scenario Accounting Engine",
                "",
                "## Key outputs",
                f"- `{out['dist_path']}`",
                f"- `{out['summary_path']}`",
                f"- `{out['sens_path']}`",
                f"- `{out['fig_path']}`",
                *([f"- `{grid_path}`"] if grid_path else []),
                "",
                "## Inputs used (latest readings)",
                f"- domestic z-score: {out['inputs']['z_domestic']}",
                f"- export YoY proxy: {out['inputs']['export_yoy_proxy']}",
                f"- property stabil prob: {out['inputs']['property_stabil_prob']}",
                "",
                "## Notes",
                "- This is **not** a time-series estimated nowcast. Parameters are explicit assumptions (defensible under limited data).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logging.info(f"[Task G] Wrote report: {rp}")

# Step H
def step_report_build(cfg: AppConfig) -> None:
    import logging
    from pathlib import Path

    spec = ReportSpec(
        repo_root=cfg.paths.repo_root,
        report_dir=Path("report"),
        report_figures_dir=Path("report/figures"),
        report_property=Path("report/report_property.md") if (cfg.paths.repo_root / "report/report_property.md").exists() else None,
        report_external=Path("report/report_external.md"),
        report_domestic=Path("report/report_domestic.md"),
        report_model=Path("report/report_model.md"),
    )

    build_main_report(spec)
    write_readme_pitch(cfg.paths.repo_root)
    logging.info("[Task H] Report build finished ✅")

STEP_REGISTRY: Dict[str, Callable[[AppConfig], None]] = {
    "smoke_test": step_smoke_test,
    "load_and_validate_data": step_load_and_validate_data,
    "build_features": step_build_features,
    "property_distribution": step_property_distribution,
    "property_dashboard_heatmap": step_property_dashboard_heatmap,
    "external_trade": step_external_trade,  
    "domestic_demand": step_domestic_demand,
    "model_scenario": step_model_scenario,
    "report_build": step_report_build,


}

def run_pipeline(cfg: AppConfig) -> None:
    cfg.paths.ensure()
    set_seed(cfg.seed)

    unknown = [s for s in cfg.steps if s not in STEP_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown steps in config.yaml: {unknown}. "
            f"Available: {sorted(STEP_REGISTRY.keys())}"
        )

    for step in cfg.steps:
        logging.info(f"== Step: {step} ==")
        STEP_REGISTRY[step](cfg)

    logging.info("Pipeline finished ✅")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="China macro quant verification pipeline (config-driven)."
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (relative to repo root or absolute).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
