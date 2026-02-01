from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .theme import COLORS, FONT_SETTINGS, apply_ib_style, annotate_latest
from src.utils.plotting import setup_mpl_chinese
setup_mpl_chinese()


def _resolve_yoy_col(df: pd.DataFrame) -> str:
    """
    Try to find the YoY column in a robust way.
    Preference order:
    1) exact matches
    2) contains 'yoy'
    3) common alternatives
    """
    candidates = ["yoy", "yoy_index", "yoy_pct", "yoy_percent", "yoy_yoy"]
    for c in candidates:
        if c in df.columns:
            return c

    # any column name containing 'yoy'
    yoy_like = [c for c in df.columns if "yoy" in str(c).lower()]
    if len(yoy_like) == 1:
        return yoy_like[0]
    if len(yoy_like) > 1:
        # prefer the one that also mentions 'index'
        for c in yoy_like:
            if "index" in str(c).lower():
                return c
        return yoy_like[0]

    raise KeyError(f"Cannot find YoY column. got columns={list(df.columns)}")


@dataclass(frozen=True)
class RegimeSpec:
    price_city_panel_csv: Path              # e.g. output/property/price_city_panel.csv
    output_dir: Path = Path("output/property")
    figures_dir: Path = Path("output/property/figures")
    low_tail_threshold: float = 95.0
    neutral_threshold: float = 100.0
    k_improve: int = 3                      # consecutive improving months for bottoming event-study
    min_cities_required: int = 50
    prob_hi: float = 0.67
    prob_lo: float = 0.33


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def _logistic(x: pd.Series) -> pd.Series:
    return 1 / (1 + np.exp(-x))


def _mad(x: pd.Series) -> float:
    arr = x.dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)))


def _monthly_distribution_features(
    df: pd.DataFrame,
    *,
    house_type: str,
    yoy_col: str,
    low_tail_threshold: float,
    neutral_threshold: float,
) -> pd.DataFrame:
    sub = df[df["house_type"] == house_type].copy()
    g = sub.groupby("date")

    out = g.apply(lambda x: pd.Series({
        "yoy_median": float(np.nanmedian(pd.to_numeric(x[yoy_col], errors="coerce").to_numpy(dtype=float))),
        "disp_mad": float(_mad(pd.to_numeric(x[yoy_col], errors="coerce"))),
        "share_pos": float((pd.to_numeric(x[yoy_col], errors="coerce") >= neutral_threshold).mean()),
        "share_deep_neg": float((pd.to_numeric(x[yoy_col], errors="coerce") <= low_tail_threshold).mean()),
        "n": int(x["city"].nunique()) if "city" in x.columns else int(pd.to_numeric(x[yoy_col], errors="coerce").notna().sum()),
    }), include_groups=False).reset_index().sort_values("date")

    # 关键：确保 house_type 被正确赋值（这一行你代码里有，但可能在 concat 时丢失了）
    out["house_type"] = house_type
    out["low_tail_threshold"] = low_tail_threshold
    out["neutral_threshold"] = neutral_threshold

    for c in ["yoy_median", "disp_mad", "share_pos", "share_deep_neg"]:
        out[f"d_{c}"] = out[c].diff()
    return out



def _compute_prob(df_feat: pd.DataFrame) -> pd.DataFrame:
    s = df_feat.sort_values("date").copy()

    components = pd.DataFrame({
        "dyoy_med_z": _zscore(s["d_yoy_median"]),      # +
        "ddisp_z": -_zscore(s["d_disp_mad"]),          # - dispersion down is good
        "dpos_z": _zscore(s["d_share_pos"]),           # +
        "dtail_z": -_zscore(s["d_share_deep_neg"]),    # - tail down is good
    })

    score = components.mean(axis=1)
    s["stabilization_score"] = score
    s["stabilization_prob"] = _logistic(score.fillna(0))
    return s


def _find_bottoming_month(sub: pd.DataFrame, k: int = 3) -> pd.Timestamp | pd.NaT:
    sub = sub.sort_values("date").copy()
    dyoy = sub["dyoy"]
    streak = np.full(len(sub), False, dtype=bool)
    for i in range(k - 1, len(sub)):
        window = dyoy.iloc[i - (k - 1): i + 1]
        streak[i] = window.notna().all() and (window > 0).all()
    sub["improving_streak"] = streak
    cand = sub[sub["improving_streak"]]
    return cand["date"].min() if not cand.empty else pd.NaT


def run_regime_signal(spec: RegimeSpec) -> Path:
    out_dir = spec.output_dir
    fig_dir = spec.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(spec.price_city_panel_csv)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

     # ---- compatibility: some city panels don't have house_type ----
    if "house_type" not in df.columns:
        if "market" in df.columns:
            # 建立映射关系
            mapping = {"new": "新房", "existing": "二手房"}
            df["house_type"] = df["market"].map(mapping).fillna(df["market"])
        else:
            df["house_type"] = "ALL"
    
    # ---- resolve YoY column name (yoy / yoy_index / etc.) ----
    yoy_col = _resolve_yoy_col(df)

    # build features for 新房/二手房
    house_types = sorted(df["house_type"].dropna().unique().tolist())
    feats = []
    for ht in house_types:
            feats.append(_monthly_distribution_features(
            df,
            house_type=str(ht),
            yoy_col=yoy_col,
            low_tail_threshold=spec.low_tail_threshold,
            neutral_threshold=spec.neutral_threshold,
        ))
    feat_all = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    if feat_all.empty:
        raise ValueError(
            f"No rows produced for regime features. "
            f"Check panel columns={list(df.columns)} and yoy/date availability."
        )

    processed_feats = []
    for ht, group in feat_all.groupby("house_type"):
        res = _compute_prob(group)
        res["house_type"] = ht  # 强制写回正确的房产类型标签
        processed_feats.append(res)
    feat_all = pd.concat(processed_feats, ignore_index=True)
    # ---- coverage gate: too few cities => do not score ----
    feat_all.loc[feat_all["n"] < spec.min_cities_required, "stabilization_prob"] = np.nan
    feat_all.loc[feat_all["n"] < spec.min_cities_required, "stabilization_score"] = np.nan

    def _label(p: float) -> str:
        if pd.isna(p):
            return "insufficient_coverage"
        if p >= spec.prob_hi:
            return "stabilizing"
        if p <= spec.prob_lo:
            return "deteriorating"
        return "transition"

    feat_all["regime_label"] = feat_all["stabilization_prob"].map(_label)

    # export table
    export_cols = [
        "date", "house_type",
        "yoy_median", "disp_mad", "share_pos", "share_deep_neg", "n",
        "stabilization_score", "stabilization_prob", "regime_label",
        "low_tail_threshold", "neutral_threshold",
    ]
    
    # 强制检查：如果 house_type 不在列里但在索引里，拉回来；如果彻底没了，报错提示
    if "house_type" not in feat_all.columns:
        if "house_type" in feat_all.index.names:
            feat_all = feat_all.reset_index()
        else:
            # 万一 apply 真的把这列弄丢了，从 index 的第一个级别尝试恢复
            feat_all = feat_all.reset_index()
            # 如果 reset 后多出了 'level_0' 这种列，重命名它（这是 Pandas 的常见坑）
            if "level_0" in feat_all.columns:
                feat_all = feat_all.rename(columns={"level_0": "house_type"})
    feat_all["house_type"] = feat_all["house_type"].astype(str)
    out_path = out_dir / "monthly_regime_table.csv"
    
    # 最后一道防线：过滤掉 DataFrame 中确实存在的列，防止 KeyError
    final_cols = [c for c in export_cols if c in feat_all.columns]
    feat_all[final_cols].sort_values(["date"]).to_csv(out_path, index=False, encoding="utf-8-sig")
    # --- [局部修改] 提前定义显示名称与目标房产类型，供后续所有绘图块引用 ---
    ht_display = {"新房": "New homes", "二手房": "Existing homes"}
    target_ht = "新房" if "新房" in house_types else (house_types[0] if house_types else "ALL")
    # ----------------------------------------------------------------------
    # -------- plot: stabilization probability
    # -------- plot: stabilization probability [Style Refactor]
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    
    # 局部硬编码色值：Midnight Blue (新房), Goldman Gold (二手房)
    color_map = {"新房": "#00204E", "二手房": "#A68B5B"}

    for house_type in house_types:
        s = feat_all[feat_all["house_type"] == house_type].sort_values("date")
        label = ht_display.get(str(house_type), str(house_type))
        ax.plot(s["date"], s["stabilization_prob"], 
                label=label, linewidth=2.5, 
                color=color_map.get(str(house_type), "#6A737B"))

    apply_ib_style(ax, 
                  title="Regime Signal: Price Stabilization Probability (0-1)", 
                  ylabel="Stabilization Probability")
    
    latest_feat = feat_all[feat_all["house_type"] == target_ht].sort_values("date")
    
    # 标注最新信号点
    target_ht_en = ht_display.get(target_ht, target_ht)
    if not latest_feat.empty:
        annotate_latest(ax, latest_feat, "stabilization_prob", f"Latest {target_ht_en}")

    ax.legend(frameon=False, loc="upper left", **FONT_SETTINGS["legend"])
    plt.tight_layout()
    plt.savefig(fig_dir / "stabilization_probability_signal.png", bbox_inches="tight")
    plt.close()

    # -------- event study: bottoming (new homes, national)
    yoy_mean = (
        df[df["house_type"].isin(house_types)]
        .groupby(["date", "house_type"], as_index=False)
        .agg(yoy_mean=(yoy_col, "mean"))
        .sort_values(["house_type", "date"])
    )
    yoy_mean["dyoy"] = yoy_mean.groupby(["house_type"])["yoy_mean"].diff()

    target_ht = "新房" if "新房" in yoy_mean["house_type"].unique() else house_types[0]
    bottom_new = _find_bottoming_month(
        yoy_mean[yoy_mean["house_type"] == target_ht],
        k=spec.k_improve,
    )

    # -------- event study: bottoming [Style Refactor]
    fig, ax = plt.subplots(figsize=(11.5, 5.5), dpi=300)
    s = yoy_mean[yoy_mean["house_type"] == target_ht].sort_values("date")
    target_label = ht_display.get(str(target_ht), str(target_ht))

    # 主趋势线：Midnight Blue
    ax.plot(s["date"], s["yoy_mean"], label=f"{target_label} | Mean YoY Index", color="#00204E", linewidth=2.5)

    if not pd.isna(bottom_new):
        # 底部垂直线：Goldman Gold
        ax.axvline(bottom_new, color="#A68B5B", linestyle="--", alpha=0.8, linewidth=1.5)
        y_top = ax.get_ylim()[1]
        ax.text(bottom_new, y_top * 0.98, " Bottoming Signal (3M consecutive improvement)", 
                va="top", fontsize=9, color="#A68B5B", fontweight='bold')

    apply_ib_style(ax, title=f"Event Study: {target_label} Mean YoY Index Bottoming Check", 
                  ylabel="YoY Index (Mean)", has_negative=False)

    ax.axhline(spec.neutral_threshold, color="#6A737B", linewidth=1, alpha=0.3, label="Neutral (100)")
    ax.legend(frameon=False, loc="lower right", **FONT_SETTINGS["legend"])
    plt.savefig(fig_dir / "event_study_bottoming_newhome.png", bbox_inches="tight")
    plt.close()


    
    # -------- proxy linkage dashboard (target house_type)
    sub = df[df["house_type"] == target_ht].copy()
    sub["_yoy"] = pd.to_numeric(sub[yoy_col], errors="coerce")

    proxy = (
        sub.groupby("date", as_index=False)
        .agg(
            share_pos=("_yoy", lambda s: float((s >= spec.neutral_threshold).mean())),
            share_deep_neg=("_yoy", lambda s: float((s <= spec.low_tail_threshold).mean())),
            yoy_median=("_yoy", lambda s: float(np.nanmedian(s.to_numpy(dtype=float)))),
            disp_mad=("_yoy", lambda s: float(_mad(s))),
            n=("_yoy", lambda s: int(s.notna().sum())),
        )
        .sort_values("date")
    )

    # -------- proxy linkage dashboard [Style Refactor]
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    # 主轴：离散度用 Slate Grey
    ax1.plot(proxy["date"], proxy["disp_mad"], label="Dispersion (MAD)", color="#6A737B", linewidth=2)
    
    apply_ib_style(ax1, title=f"Proxy Linkage: {target_ht} Dispersion vs Breadth & Tail", ylabel="Dispersion (MAD)")

    ax2 = ax1.twinx()
    # 覆盖率：Goldman Gold (Breadth) 和 Alert Red (Tail Pressure)
    ax2.plot(proxy["date"], proxy["share_pos"] * 100, linestyle="--", color="#A68B5B", 
             label=f"YoY≥{spec.neutral_threshold:g} (Breadth %)", linewidth=2, alpha=0.85)
    ax2.plot(proxy["date"], proxy["share_deep_neg"] * 100, linestyle="--", color="#9E1B32", 
             label=f"YoY≤{spec.low_tail_threshold:g} (Tail %)", linewidth=2, alpha=0.85)
    
    ax2.set_ylabel("Share of Cities (%)", **FONT_SETTINGS["label"])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right", **FONT_SETTINGS["legend"])

    plt.tight_layout()
    plt.savefig(fig_dir / "proxy_linkage_dashboard.png", bbox_inches="tight")
    plt.close()

    return out_path
