from __future__ import annotations

from pathlib import Path
import argparse
import yaml

from src.property.housing_clean import CleanSpec, build_panel
from src.property.heatmap_dashboard import HeatmapSpec, make_dashboard


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_cfg() -> dict:
    cfg_path = _repo_root() / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return cfg.get("property", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["clean", "heatmap", "all"], default="all")
    args = p.parse_args()

    cfg = _load_cfg()
    root = _repo_root()

    price_xlsx = root / cfg.get("price_xlsx_path", "data/raw/housing price.xlsx")
    panel_csv  = root / cfg.get("panel_csv_path", "output/property/processed/housing_price_panel.csv")
    heatmap_png = root / cfg.get("heatmap_png_path", "output/property/figures/yoy_recovery_dashboard_heatmap_with_delta.png")

    start_date = cfg.get("start_date", "2025-01-01")
    row_gap = float(cfg.get("row_gap", 1.4))

    if args.task in ("clean", "all"):
        panel_csv.parent.mkdir(parents=True, exist_ok=True)
        spec = CleanSpec(price_xlsx_path=price_xlsx, out_csv_path=panel_csv)
        df_panel = build_panel(spec)
        df_panel.to_csv(panel_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] clean -> {panel_csv}")

    if args.task in ("heatmap", "all"):
        heatmap_png.parent.mkdir(parents=True, exist_ok=True)
        spec = HeatmapSpec(panel_csv_path=panel_csv, out_png_path=heatmap_png, start_date=start_date, row_gap=row_gap)
        import pandas as pd
        df = pd.read_csv(panel_csv)
        df["date"] = pd.to_datetime(df["date"])
        make_dashboard(df, spec)
        print(f"[OK] heatmap -> {heatmap_png}")


if __name__ == "__main__":
    main()
