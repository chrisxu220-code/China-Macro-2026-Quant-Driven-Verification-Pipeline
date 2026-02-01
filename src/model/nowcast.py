from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .scenario import ScenarioSpec, run_scenario_engine
from .metrics import summarize_distribution, to_one_row_df
from .theme import COLORS, FONT_SETTINGS, apply_ib_style


# -----------------------------
# Spec
# -----------------------------
@dataclass(frozen=True)
class AnchorLine:
    """
    Vertical reference line on distribution plots.
    value is in growth terms (e.g., 0.048 = 4.8%).
    """
    label: str
    value: float
    linestyle: str = "--"
    linewidth: float = 1.8
    alpha: float = 0.9


@dataclass(frozen=True)
class ModelSpec:
    demand_mix_panel_csv: Path
    trade_panel_csv: Path
    property_regime_csv: Path | None = None

    output_dir: Path = Path("output/model")
    figures_dir: Path = Path("output/model/figures")

    baseline_growth: float = 0.045
    beta_domestic: float = 0.0030
    beta_property: float = 0.0020
    beta_external: float = 0.0015

    n_sims: int = 2000
    seed: int = 42
    param_sigma_scale: float = 0.25

    # NEW: external benchmarks / anchors (IMF, GS, etc.)
    anchors: Optional[Sequence[AnchorLine]] = None


# -----------------------------
# Plotting
# -----------------------------
def _kde_line(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Lightweight KDE (no scipy/seaborn).
    Gaussian kernel with Silverman bandwidth.
    """
    x = x.astype(float)
    n = x.size
    if n < 2:
        return np.zeros_like(grid)

    std = float(np.std(x, ddof=1))
    if std == 0:
        return np.zeros_like(grid)

    h = 1.06 * std * (n ** (-1 / 5))  # Silverman
    if h <= 0:
        return np.zeros_like(grid)

    z = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))
    return dens

def _plot_distribution(
    dist_csv: Path,
    out_path: Path,
    title: str,
    anchors: Optional[Sequence[AnchorLine]] = None,
) -> dict:
    """
    Refactored with Buy-side Visual Identity System (VIS)
    """
    df = pd.read_csv(dist_csv)
    x = pd.to_numeric(df["growth"], errors="coerce").dropna().to_numpy()
    if x.size == 0:
        raise ValueError(f"No valid 'growth' values found in {dist_csv}")

    mean_val = float(np.mean(x))
    p10, p50, p90 = [float(np.percentile(x, q)) for q in (10, 50, 90)]

    # 1. 初始化画布 (应用 300 DPI 投行标准)
    fig, ax = plt.subplots(figsize=(11.2, 5.8), dpi=300)

    # 2. 绘制直方图：使用浅灰色作为底色，突出关键线条
    ax.hist(
        x,
        bins=34,
        density=True,
        alpha=0.6,
        edgecolor="white",
        color="#D1D5D8",  # 对应 COLORS["grid"]
        zorder=1
    )

    # 3. P10–P90 置信区间带：使用“高盛金”填充
    ax.axvspan(p10, p90, alpha=0.15, color="#A68B5B", 
               label=f"P10–P90 Band ({p10*100:.1f}%–{p90*100:.1f}%)", zorder=2)

    # 4. 模型均值线：使用“深夜蓝”加粗实线
    ax.axvline(
        mean_val,
        color="#00204E",
        linestyle="-",
        linewidth=3.0,
        alpha=1.0,
        label=f"Model Mean ({mean_val*100:.2f}%)",
        zorder=5,
    )

    # 5. 外部参考锚点（IMF, GS 等）：使用“板岩灰”虚线
    seen = set()
    for a in (anchors or []):
        key = (a.label, float(a.value))
        if key in seen: continue
        seen.add(key)

        ax.axvline(
            a.value,
            color="#6A737B",
            linestyle=a.linestyle,
            linewidth=1.5,
            alpha=0.8,
            label=f"{a.label} ({a.value*100:.1f}%)",
            zorder=4,
        )

    # 6. 应用全局样式模板
    apply_ib_style(ax, title=title, ylabel="Probability Density")
    ax.set_xlabel("Projected Growth Rate (2026E)", **FONT_SETTINGS["label"])
    
    # 7. 优化图例
    ax.legend(loc="upper left", frameon=False, **FONT_SETTINGS["legend"])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return {"mean": mean_val, "p10": p10, "p50": p50, "p90": p90}



# -----------------------------
# Main
# -----------------------------
def run_model(spec: ModelSpec) -> dict:
    logging.info("[Task G] Running model spec...")

    scen = ScenarioSpec(
        demand_mix_panel_csv=spec.demand_mix_panel_csv,
        trade_panel_csv=spec.trade_panel_csv,
        property_regime_csv=spec.property_regime_csv,
        output_dir=spec.output_dir,
        figures_dir=spec.figures_dir,
        baseline_growth=spec.baseline_growth,
        beta_domestic=spec.beta_domestic,
        beta_property=spec.beta_property,
        beta_external=spec.beta_external,
        n_sims=spec.n_sims,
        seed=spec.seed,
        param_sigma_scale=spec.param_sigma_scale,
    )

    res = run_scenario_engine(scen)

    spec.figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = spec.figures_dir / "growth_distribution.png"

    # Default anchors: baseline + optional IMF/GS etc. from spec.anchors
    anchors = []
    anchors.append(AnchorLine(label="Baseline (policy-neutral)", value=spec.baseline_growth, linestyle="--", linewidth=1.8, alpha=0.75))

    if spec.anchors:
        anchors.extend(list(spec.anchors))

    stats = _plot_distribution(
        res["dist_path"],
        fig_path,
        "Growth distribution (scenario accounting engine)",
        anchors=anchors,
    )

    # summary stats
    df = pd.read_csv(res["dist_path"])
    x = pd.to_numeric(df["growth"], errors="coerce").dropna().to_numpy()
    summ = summarize_distribution(x)
    summ_df = to_one_row_df(summ, label="scenario_engine")
    summ_path = spec.output_dir / "growth_summary.csv"
    summ_df.to_csv(summ_path, index=False, encoding="utf-8-sig")

    # also write a small “benchmarks” csv so the report has provenance
    bench_path = spec.output_dir / "growth_benchmarks.csv"
    bench_rows = [{"label": a.label, "value": a.value} for a in anchors]
    pd.DataFrame(bench_rows).to_csv(bench_path, index=False, encoding="utf-8-sig")

    return {
        **res,
        "fig_path": fig_path,
        "summary_path": summ_path,
        "benchmarks_path": bench_path,
        "plot_stats": stats,
    }
