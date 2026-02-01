from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap

# =========================
# 字体（保持你现在的设定）
# =========================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class HeatmapSpec:
    panel_csv_path: Path
    out_png_path: Path
    start_date: str = "2025-01-01"
    row_gap: float = 1.4

    # labels
    title: str = "YoY Recovery Heatmap Dashboard | New homes vs Existing homes (sorted by ΔYoY)"
    cbar_label: str = "YoY index (winsorized p5–p95)"
    delta_title: str = "ΔYoY (last − first)"
    share_pos: str = "Share > 0"


WHITELIST = ["北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "合肥", "西安"]
WHITELIST_SET = set(WHITELIST)
SHOW_WHITELIST = False

HOUSE_EN = {"新房": "New homes", "二手房": "Existing homes"}

CITY_EN = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "成都": "Chengdu",
    "重庆": "Chongqing",
    "合肥": "Hefei",
    "西安": "Xi'an",
    "金华": "Jinhua",
    "泉州": "Quanzhou",
    "宁波": "Ningbo",
    "南宁": "Nanning",
    "温州": "Wenzhou",
    "宜昌": "Yichang",
    "长沙": "Changsha",
    "韶关": "Shaoguan",
    "福州": "Fuzhou",
    "天津": "Tianjin",
    "北海": "Beihai",
    "包头": "Baotou",
    "南充": "Nanchong",
    "桂林": "Guilin",
    "海口": "Haikou",
    "太原": "Taiyuan",
    "牡丹江": "Mudanjiang",
    "平顶山": "Pingdingshan",
    "丹东": "Dandong",
    "西宁": "Xi'ning",
    "呼和浩特": "Hohhot",
    "济南": "Jinan",
    "锦州": "Jinzhou",
    "南昌": "Nanchang",
    "银川": "Yinchuan",
    "郑州": "Zhengzhou",
    "济宁": "Jining",
    "徐州": "Xuzhou",
    "南京": "Nanjing",
}


def city_label(c: str) -> str:
    return CITY_EN.get(c, c)


def build_pivot_and_delta(df_all: pd.DataFrame, house_type: str, start_date: str) -> tuple[pd.DataFrame, pd.Series]:
    sub = df_all[(df_all["house_type"] == house_type) & (df_all["date"] >= start_date)].copy()
    pivot = sub.pivot_table(index="city", columns="date", values="yoy").sort_index(axis=1)

    # 只保留覆盖较好的月份：>=50 城有值（沿用你现在逻辑）
    valid_cols = [c for c in pivot.columns if pivot[c].notna().sum() >= 50]
    pivot = pivot[valid_cols]

    if pivot.shape[1] < 2:
        raise ValueError(f"{house_type} 可用月份太少（<2）。检查数据覆盖或调整 start_date。")

    first_col, last_col = pivot.columns[0], pivot.columns[-1]
    delta = pivot[last_col] - pivot[first_col]

    # 按 recovery 排序
    order = delta.sort_values(ascending=False).index
    pivot = pivot.loc[order]
    delta = delta.loc[order]
    return pivot, delta


def make_dashboard(df: pd.DataFrame, spec: HeatmapSpec) -> None:
    pivot_new, delta_new = build_pivot_and_delta(df, "新房", spec.start_date)
    pivot_sec, delta_sec = build_pivot_and_delta(df, "二手房", spec.start_date)

    # unified color scale
    all_vals = np.concatenate([pivot_new.to_numpy().ravel(), pivot_sec.to_numpy().ravel()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = np.quantile(all_vals, 0.05)
    vmax = np.quantile(all_vals, 0.95)
    gs_colors = ["#F7F7F7", "#D4C4A8", "#A68B5B", "#6B5533"]
    cmap = LinearSegmentedColormap.from_list("gs_heatmap", gs_colors, N=256)
    cmap.set_bad(color="#E6E6E6") # 缺失值保持浅灰

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        width_ratios=[2.8, 14.7, 3, 0.5],
        hspace=0.25, wspace=0.1
    )
    cax = fig.add_subplot(gs[:, 3])

    def plot_one_panel(pivot: pd.DataFrame, delta: pd.Series, row: int, house_type: str, show_xlabels: bool):
        ax_hm  = fig.add_subplot(gs[row, 1])
        ax_lbl = fig.add_subplot(gs[row, 0], sharey=ax_hm)
        ax_bar = fig.add_subplot(gs[row, 2], sharey=ax_hm)

        cities = list(pivot.index)
        n_cities = len(cities)
        y_pos = np.arange(n_cities) * spec.row_gap

        im = ax_hm.imshow(
            pivot.to_numpy(),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=[0, pivot.shape[1], y_pos[-1] + spec.row_gap/2, -spec.row_gap/2]
        )

        dates = list(pivot.columns)
        if show_xlabels:
            n_dates = len(dates)
            step = max(1, n_dates // 12)
            ax_hm.set_xticks(np.arange(0, n_dates, step))
            ax_hm.set_xticklabels(
                [dates[i].strftime("%Y-%m") for i in range(0, n_dates, step)],
                rotation=45, ha="right", fontsize=9
            )
        else:
            ax_hm.tick_params(axis="x", bottom=False, labelbottom=False)

        ax_hm.tick_params(axis="y", left=False, labelleft=False)
        house_en = HOUSE_EN.get(house_type, house_type)
        ax_hm.set_title(
            f"{house_en} | YoY recovery heatmap ({dates[0].strftime('%Y-%m')}–{dates[-1].strftime('%Y-%m')})",
            fontsize=11, pad=6
        )

        # left labels
        ax_lbl.set_xlim(0, 1)
        ax_lbl.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for spine in ax_lbl.spines.values():
            spine.set_visible(False)

        top_k, bot_k = 10, 10
        top_indices = list(range(0, min(top_k, n_cities)))
        bot_indices = list(range(max(0, n_cities - bot_k), n_cities))

        def place_group(indices, y_text_min, y_text_max, fontsize=8):
            if not indices:
                return
            ys = np.linspace(y_text_min, y_text_max, len(indices))
            for i, y_text in zip(indices, ys):
                y_row = y_pos[i]
                label = city_label(cities[i])

                ax_lbl.text(
                    0.98, y_text, label,
                    transform=ax_lbl.get_yaxis_transform(),  # x=axes, y=data
                    ha="right", va="center", fontsize=fontsize
                )

                # 线指到 heatmap 左边缘（你最终想要的效果）
                con = ConnectionPatch(
                    xyA=(1.00, y_text), coordsA=ax_lbl.get_yaxis_transform(),
                    xyB=(0.0,  y_row),  coordsB=ax_hm.transData,
                    color="gray", alpha=0.35, lw=0.7
                )
                fig.add_artist(con)

        TOP_SPAN = 28
        BOT_SPAN = 28
        place_group(top_indices, y_pos[0], y_pos[min(TOP_SPAN, n_cities-1)], fontsize=8)
        place_group(bot_indices, y_pos[max(0, n_cities-1-BOT_SPAN)], y_pos[-1], fontsize=8)

        if SHOW_WHITELIST:
            whitelist_idx = [i for i, c in enumerate(cities) if (c in WHITELIST_SET) and (i not in set(top_indices + bot_indices))]
            if whitelist_idx:
                mid_start = y_pos[min(TOP_SPAN, n_cities - 1)] + spec.row_gap * 1.5
                mid_end   = y_pos[max(0, n_cities - 1 - BOT_SPAN)] - spec.row_gap * 1.5
                if mid_end > mid_start:
                    place_group(whitelist_idx, mid_start, mid_end, fontsize=9)

        # 统一使用你的深夜蓝作为增量条的颜色
        ax_bar.barh(y_pos, delta.values, height=0.8 * spec.row_gap, color="#00204E", alpha=0.8)
        ax_bar.axvline(0, color="black", linewidth=0.8, alpha=0.5)

        share_pos = (delta > 0).mean() * 100
        ax_bar.set_title(f"{spec.delta_title}\n{spec.share_pos}: {share_pos:.0f}%", fontsize=10)
        ax_bar.grid(axis="x", linestyle="--", alpha=0.3)
        ax_bar.tick_params(axis="y", left=False, labelleft=False)

        ax_hm.set_ylim(y_pos[-1] + spec.row_gap/2, -spec.row_gap/2)
        ax_lbl.set_ylim(ax_hm.get_ylim())
        ax_bar.set_ylim(ax_hm.get_ylim())
        return im

    im0 = plot_one_panel(pivot_new, delta_new, row=0, house_type="新房", show_xlabels=False)
    _   = plot_one_panel(pivot_sec, delta_sec, row=1, house_type="二手房", show_xlabels=True)

    fig.suptitle(spec.title, fontsize=16, y=0.98)
    cbar = fig.colorbar(im0, cax=cax)
    cbar.set_label(spec.cbar_label)

    spec.out_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(spec.out_png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[OK] wrote: {spec.out_png_path}")


def main():
    repo_root = Path(__file__).resolve().parents[2]

    default_panel = repo_root / "output" / "property" / "processed" / "housing_price_panel.csv"
    default_png   = repo_root / "output" / "property" / "figures" / "yoy_recovery_dashboard_heatmap_with_delta.png"

    spec = HeatmapSpec(panel_csv_path=default_panel, out_png_path=default_png)

    df = pd.read_csv(spec.panel_csv_path)
    df["date"] = pd.to_datetime(df["date"])
    make_dashboard(df, spec)


if __name__ == "__main__":
    main()
