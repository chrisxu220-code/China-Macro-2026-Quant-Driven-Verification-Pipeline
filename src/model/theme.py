from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ==========================================
# Buy-Side Visual Identity System (VIS)
# ==========================================

# 投行专用色板
COLORS = {
    "primary": "#00204E",     # Midnight Blue (主趋势线)
    "secondary": "#6A737B",   # Slate Grey (次要参考线)
    "highlight": "#A68B5B",   # Goldman Gold (领先指标/新动能)
    "risk": "#9E1B32",        # Alert Red (风险/旧经济)
    "grid": "#D1D5D8",        # 极浅灰色 (网格)
    "white": "#FFFFFF"
}

# 字体规范
FONT_SETTINGS = {
    "title": {"fontsize": 14, "fontweight": "bold", "family": "sans-serif"},
    "label": {"fontsize": 10, "family": "sans-serif"},
    "legend": {"fontsize": 9},
    "tick": {"labelsize": 9}
}

def apply_ib_style(ax: plt.Axes, title: str, ylabel: str, has_negative: bool = False):
    """
    一键应用买方研报视觉标准
    """
    # 设置标题 (左对齐，这是顶级投行研报的标准写法)
    ax.set_title(title, loc='left', **FONT_SETTINGS["title"], pad=20)
    ax.set_ylabel(ylabel, **FONT_SETTINGS["label"])
    
    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    
    # 仅保留轻微的水平网格线
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', color=COLORS["secondary"])
    ax.xaxis.grid(False)
    
    # 刻度线与日期格式化
    ax.tick_params(axis='both', which='major', **FONT_SETTINGS["tick"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 如果有负值，加粗 y=0 基准线
    if has_negative:
        ax.axhline(0, color=COLORS["primary"], linewidth=0.8, zorder=1)

def annotate_latest(ax: plt.Axes, df: pd.DataFrame, y_col: str, label: str = "Latest"):
    """
    为最新数据点添加标注框
    """
    if df.empty:
        return
    latest = df.iloc[-1]
    ax.annotate(
        f"{label}: {latest[y_col]:.2f}",
        xy=(latest["date"], latest[y_col]),
        xytext=(15, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["highlight"], alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=COLORS["highlight"]),
        fontsize=8, 
        fontweight='bold'
    )