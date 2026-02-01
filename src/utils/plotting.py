from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# --- 投行标准视觉规范 (Single Source of Truth) ---
IB_THEME = {
    "colors": {
        "primary": "#00204E",     # Midnight Blue
        "highlight": "#A68B5B",   # Goldman Gold
        "secondary": "#6A737B",   # Slate Grey
        "risk": "#9E1B32",        # Alert Red
        "grid": "#D1D5D8"
    },
    "cycle": ["#00204E", "#A68B5B", "#6A737B", "#9E1B32"]
}

def setup_mpl_chinese() -> None:
    """全局初始化：处理字体和基础配色循环"""
    candidates = [
        "PingFang SC", "Heiti SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"
    ]
    
    # 注入基础配置
    mpl.rcParams["font.sans-serif"] = candidates + mpl.rcParams.get("font.sans-serif", [])
    mpl.rcParams["axes.unicode_minus"] = False 
    mpl.rcParams["font.family"] = "sans-serif"
    
    # 强制设置颜色循环
    mpl.rcParams["axes.prop_cycle"] = cycler(color=IB_THEME["cycle"])
    
    # 导出高清设置
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["figure.dpi"] = 120

def get_ib_cycler():
    """用于在局部绘图时强制调用颜色循环"""
    return cycler(color=IB_THEME["cycle"])