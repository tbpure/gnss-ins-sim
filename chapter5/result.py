import numpy as np
import pandas as pd

from chapter5.plot_utils import plot_rmse_bars, reformat_by_direction
from utils.plot_utils import *

length = 30
# 原始数据

dfs = pd.read_excel("concat_data.xlsx", sheet_name=["North", "East", "Height"])

df_N = dfs["North"]
df_E = dfs["East"]
df_U = dfs["Height"]

cols = df_N.columns
if "t" in cols or "time" in cols:
    cols = cols[1:]


data = []
ref = np.zeros(1)
labels = []

exclude_columns = {'ins', 'pred-original', 'ekf-original', 'akf-original'}
for col in cols:
    if(col in exclude_columns):
        continue
    if(col == "ref"):
        ref = np.stack([
        df_N[col].values,
        df_E[col].values,
        df_U[col].values
    ], axis=1)[1:length + 1, :]
        continue
    traj = np.stack([
        df_N[col].values,
        df_E[col].values,
        df_U[col].values
    ], axis=1)[1:length + 1, :]

    data.append(traj)
    labels.append(col)

errors = []
errors_abs = []
for algo in data:
    error = algo - ref
    errors_abs.append(np.abs(error))
    errors.append(error)

n, e, h, labels = reformat_by_direction(errors_abs, labels)

labels[0] = "CNN_SEGGRU"
labels = [x.upper() for x in labels]
import matplotlib.pyplot as plt
import numpy as np

# 1. 计算水平方向误差 (Horizontal = sqrt(North^2 + East^2))
# 假设 n, e 是包含多个算法误差数据的列表或数组
horiz = []
for i in range(len(n)):
    horiz.append(np.sqrt(n[i] ** 2 + e[i] ** 2))

# 2. 设置绘图参数
# 你可以根据需要调整 figsize


# 定义每个方向的数据和对应的 Y 轴标签
plot_configs = [
    {'data': n, 'ylabel': 'North APE/m'},
    {'data': e, 'ylabel': 'East APE/m'},
    {'data':h, 'ylabel': 'Vertical APE/m'},
    {'data': horiz, 'ylabel': 'Horizontal APE/m'},
]
fig, axes = plt.subplots(len(plot_configs), 1, figsize=(12, 10), sharex=True)
plt.subplots_adjust(hspace=0.2)  # 调整子图间距

# 3. 循环绘制三个子图
for i, config in enumerate(plot_configs):
    ax = axes[i]

    # 调用你本地的绘图函数
    # 注意：这里假设你的 plot_rmse_bars 接受 (ax, data, labels) 等参数
    # 如果该函数不支持传入 ax，你可能需要根据 plot_rmse_bars 的内部实现进行调整
    plot_rmse_bars(
        ax=ax,
        data=config['data'],
        labels=labels,
        # 这里的 color 可以对应图片中的 橙色、黄色、青色
        colors=['#E17A5D', '#E9C46A', '#2A9D8F']
    )

    # 设置细节
    ax.set_ylabel(config['ylabel'], fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='-', alpha=0.3)
    if(config['ylabel'] != "Vertical APE/m"):
        ax.set_ylim(0, 800)  # 根据原图设置 Y 轴刻度范围

    # 设置 X 轴刻度 (1-30)
    ax.set_xticks(range(len(config['data'][0])))
    ax.set_xticklabels(range(1, len(config['data'][0]) + 1))

# 4. 设置最下方的 X 轴标签
axes[-1].set_xlabel('APE', fontsize=12, fontweight='bold')

# 5. 保存或展示
plt.tight_layout()
plt.show()