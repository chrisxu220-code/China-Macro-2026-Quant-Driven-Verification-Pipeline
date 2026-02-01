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

# 在 theme.py 中修改函数定义
def apply_ib_style(ax, title=None, ylabel=None, has_negative=False, is_timeseries=True):
    """
    应用投行风格样式
    is_timeseries: 如果为 True，则格式化 X 轴为日期；如果为 False，则保持普通数值。
    """
    # ... 现有的样式代码 (如 ax.spines["top"].set_visible(False) 等) ...
    
    if title:
        ax.set_title(title, **FONT_SETTINGS["title"])
    if ylabel:
        ax.set_ylabel(ylabel, **FONT_SETTINGS["label"])

    # 关键修改：只有当 is_timeseries 为 True 时才应用日期格式化
    if is_timeseries:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        # 如果不是时间序列，确保它是普通数字格式
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())

    if has_negative:
        ax.axhline(0, color="#6A737B", linewidth=0.8, zorder=1)

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