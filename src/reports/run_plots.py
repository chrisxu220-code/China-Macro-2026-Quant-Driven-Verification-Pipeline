from __future__ import annotations

import argparse
import logging

from ..config import load_config
from .plots import PlotPaths, run_all_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate research figures (Task D).")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # keep logging simple for this runner
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    paths = PlotPaths(
        features_dir=cfg.paths.output_dir / "features",
        figures_dir=cfg.paths.figures_dir,
    )
    run_all_plots(paths)
    logging.info("[Task D] Figures finished ✅")


if __name__ == "__main__":
    main()
