import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ==========================================
# 模拟你的 VIS 环境（防止 import 失败）
# ==========================================
COLORS = {"primary": "#00204E", "secondary": "#6A737B", "highlight": "#A68B5B"}
FONT_SETTINGS = {"title": {"fontsize": 14, "fontweight": "bold"}, "label": {"fontsize": 10}}

def apply_ib_style_test(ax, title, is_timeseries=True):
    ax.set_title(title, loc='left', **FONT_SETTINGS["title"], pad=20)
    if not is_timeseries:
        # 【物理重置】强行断开日期轴的连接
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# ==========================================
# 核心测试逻辑
# ==========================================
def debug_plot():
    # 1. 直接读取你那个中间表
    csv_path = "output/property/price_sales_inventory_validation_table.csv"
    try:
        df = pd.read_csv(csv_path)
    except:
        print(f"找不到文件: {csv_path}，请确认路径！")
        return

    # 2. 模拟计算相关性 (Lag -3, 0, 3)
    lags = [-3, 0, 3]
    results = []
    x = pd.to_numeric(df["stabilization_prob"])
    y = pd.to_numeric(df["sales_area"])
    
    for k in lags:
        yy = y.shift(-k)
        mask = x.notna() & yy.notna()
        corr = x[mask].corr(yy[mask])
        results.append({"lag": k, "corr": corr})
    
    res_df = pd.DataFrame(results)
    print("测试数据计算结果:\n", res_df)

    # 3. 绘图 - 采用最极端的清空模式
    plt.close('all') 
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 使用 numpy 数组绘图，彻底甩掉 Pandas Index 的元数据干扰
    ax.plot(res_df["lag"].values, res_df["corr"].values, 
            marker='o', color=COLORS["highlight"], linewidth=2)
    
    # 强制关闭时间序列并刷新格式
    apply_ib_style_test(ax, "DEBUG TEST: Lead-Lag Corr", is_timeseries=False)
    
    # 设置显示范围，防止被“幽灵日期”撑大
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.1, 1.1)
    
    test_out = "test_debug_plot.png"
    plt.savefig(test_out, dpi=200)
    print(f"测试图已生成: {test_out}")

if __name__ == "__main__":
    debug_plot()