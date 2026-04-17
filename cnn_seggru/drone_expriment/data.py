import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# ======================
# 配置与样式设定 (IEEE Standard)
# ======================


# 设置英文字体（IEEE标准）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 指定中文字体（Mac推荐）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 可选：'Heiti TC', 'STHeiti', 'Arial Unicode MS'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "font.family": ['Times New Roman', 'Arial Unicode MS'],  # ⭐核心

    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",

    "axes.unicode_minus": False
})
# 算法配置：颜色、线型、权重
# 将提出的算法放在显眼位置
algo_configs = {
    'orange': {
        'label': '真实轨迹 (Ground Truth)',
        'color': '#6E6E6E',
        'ls': '--',
        'lw': 2.0,
        'z': 20,
        'alpha': 0.9
    },

    'blue': {
        'label': 'CNN-SEGGRU',
        'color': '#4C72B0',      # 高级蓝（替代刺眼红）
        'ls': '-',
        'lw': 2.4,               # 主模型最粗
        'z': 18
    },

    'red': {
        'label': 'IMU原始解',
        'color': '#B0B0B0',      # 更浅灰（弱化噪声方法）
        'ls': ':',
        'lw': 1.2,
        'z': 5
    },

    'yellow': {
        'label': 'GRU方法',
        'color': '#55A868',      # 绿色（与主蓝形成冷暖对比）
        'ls': '--',
        'lw': 1.8,
        'z': 10
    },

    'purple': {
        'label': 'CNN-GRU方法',
        'color': '#C44E52',      # 柔和红（作为次强对比）
        'ls': '-.',
        'lw': 1.8,
        'z': 12
    }
}

directions = {
    'n': '北向 (North)',
    'e': '东向 (East)',
    'h': '天向 (Up)'
}

rmse_results = {d: {} for d in directions}
# ======================
# 数据处理函数
# ======================
def load_and_average(file_path, window=0.5):
    try:
        df = pd.read_csv(file_path, header=None)
        x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        y *= 10
        x_new = np.arange(0, 61, 1)
        y_new = []
        for xi in x_new:
            mask = (x >= xi - window) & (x < xi + window)
            y_subset = y[mask]
            y_new.append(np.mean(y_subset) if len(y_subset) > 0 else np.nan)

        y_new = np.array(y_new)
        nan_mask = np.isnan(y_new)
        if nan_mask.any():
            y_new[nan_mask] = np.interp(x_new[nan_mask], x_new[~nan_mask], y_new[~nan_mask])
        return x_new, y_new
    except FileNotFoundError:
        return None, None


# ======================
# 绘图逻辑优化
# ======================

# 预加载数据，避免重复读取
data_cache = {}
for d_code in directions:
    data_cache[d_code] = {}
    for color in algo_configs:
        path = f'datas/{d_code}-{color}.csv'
        data_cache[d_code][color] = load_and_average(path)

# ---------- 图1：轨迹对比 ----------
fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))

for idx, (d_code, d_name) in enumerate(directions.items()):
    ax = axes1[idx]
    for color, cfg in algo_configs.items():
        x, y = data_cache[d_code][color]
        if x is None: continue

        ax.plot(x, y, label=cfg['label'], color=cfg['color'],
                linestyle=cfg['ls'], linewidth=cfg['lw'], zorder=cfg['z'])

    ax.set_title(f"{d_name} 位置")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("位置 (m)")
# 获取图例句柄
handles, labels = axes1[0].get_legend_handles_labels()

# 关键：先执行 tight_layout，并为顶部留出空间 (rect 参数)
# rect=[左, 下, 右, 上]
fig1.tight_layout(rect=[0, 0, 1, 0.92])

# 将图例放在留出的空白区域 (y=0.95 左右)
fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=5, frameon=False, fontsize=10)



# ---------- 图2：误差与统计 ----------
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

for idx, (d_code, d_name) in enumerate(directions.items()):
    ax = axes2[idx]
    x_ref, y_ref = data_cache[d_code]['orange']

    for color, cfg in algo_configs.items():
        if color == 'orange': continue

        x, y = data_cache[d_code][color]
        if x is None: continue

        error = y - y_ref
        rmse = np.sqrt(np.mean(error ** 2))
        rmse_results[d_code][color] = rmse
        # 针对你提出的算法，可以在标签里加个特殊记号或者加粗（如果环境支持）
        label_text = f"{cfg['label']}（RMSE: {rmse:.2f} m）"

        ax.plot(x, error, label=label_text, color=cfg['color'],
                linestyle=cfg['ls'], linewidth=cfg['lw'], zorder=cfg['z'])

        # if "本文方法" in cfg['label']:
        #     ax.plot(x, error, label=label_text,
        #             color=cfg['color'], linestyle=cfg['ls'],
        #             linewidth=2.5, zorder=15)

    ax.set_title(f"{d_name} 误差分析")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("误差 (m)")
    ax.axhline(0, color='black', lw=1, alpha=0.5)


# 同样的处理方式
handles2, labels2 = axes2[0].get_legend_handles_labels()
fig2.tight_layout(rect=[0, 0, 1, 0.88]) # 误差图图例较长，多留一点空间

fig2.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=3, frameon=False, fontsize=9)

if (1 == 0):
    fig1.savefig("result/trajectory_comparison.pdf", bbox_inches='tight')
    fig1.savefig("result/trajectory_comparison.png", dpi=600, bbox_inches='tight')
    fig2.savefig("result/error_comparison.pdf", bbox_inches='tight')
    fig2.savefig("result/error_comparison.png", dpi=600, bbox_inches='tight')

plt.show()


# ======================
# RMSE结果汇总
# ======================
rmse_table = pd.DataFrame(rmse_results).T  # 行=方向，列=算法

# 去掉 Ground Truth
if 'orange' in rmse_table.columns:
    rmse_table = rmse_table.drop(columns=['orange'])

# 重命名列（更论文风格）
rename_map = {k: v['label'] for k, v in algo_configs.items() if k != 'orange'}
rmse_table = rmse_table.rename(columns=rename_map)

# 保留两位小数
rmse_table = rmse_table.round(2)

print("\n===== RMSE Results (m) =====")
print(rmse_table)