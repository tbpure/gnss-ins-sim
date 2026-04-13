import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chapter5.plot_utils import plot_rmse_bars, reformat_by_direction

# ======================
# 参数
# ======================
length = 30

# ======================
# 读取数据
# ======================
dfs = pd.read_excel(
    "datas/concat_data.xlsx",
    sheet_name=["North", "East", "Height"]
)

df_N = dfs["North"]
df_E = dfs["East"]
df_U = dfs["Height"]

cols = df_N.columns

# 去掉时间列
if "t" in cols or "time" in cols:
    cols = cols[1:]

# ======================
# 构建轨迹数据
# ======================
data = []
labels = []
ref = None

exclude_columns = {'ins', 'pred-original', 'ekf-original', 'akf-original'}

for col in cols:
    if col in exclude_columns:
        continue

    traj = np.stack([
        df_N[col].values,
        df_E[col].values,
        df_U[col].values
    ], axis=1)[1:length + 1, :]

    if col == "ref":
        ref = traj
    else:
        data.append(traj)
        labels.append(col)

# ======================
# 计算误差
# ======================
errors_abs = []

for algo in data:
    error = algo - ref
    errors_abs.append(np.abs(error))

# ======================
# 按方向拆分
# ======================
n, e, h, labels = reformat_by_direction(errors_abs, labels)

labels[0] = "CNN_SEGGRU"
labels = [x.upper() for x in labels]

# ======================
# RMSE柱状图 + 水平误差
# ======================
horiz = [
    np.sqrt(n[i] ** 2 + e[i] ** 2)
    for i in range(len(n))
]

plot_configs = [
    {'data': n, 'ylabel': 'North APE/m'},
    {'data': e, 'ylabel': 'East APE/m'},
    {'data': h, 'ylabel': 'Vertical APE/m'},
    {'data': horiz, 'ylabel': 'Horizontal APE/m'},
]

fig, axes = plt.subplots(len(plot_configs), 1, figsize=(12, 10), sharex=True)
plt.subplots_adjust(hspace=0.25)

for i, config in enumerate(plot_configs):
    ax = axes[i]

    plot_rmse_bars(
        ax=ax,
        data=config['data'],
        labels=labels
    )

    ax.set_ylabel(config['ylabel'], fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='-', alpha=0.3)

    if config['ylabel'] != "Vertical APE/m":
        ax.set_ylim(0, 800)

    ax.set_xticks(range(len(config['data'][0])))
    ax.set_xticklabels(range(1, len(config['data'][0]) + 1))

axes[-1].set_xlabel('Time Step', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ======================
# ⭐ 轨迹投影图（XY / XZ / YZ）
# ======================
all_trajs = [ref] + data
all_labels = ["REF"] + labels


def plot_projection(ax, trajs, labels, idx1, idx2, xlabel, ylabel, title):

    all_x = []
    all_y = []

    for traj in trajs:
        all_x.append(traj[:, idx1])
        all_y.append(traj[:, idx2])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    # ===== 自适应范围 =====
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # ===== 画轨迹 =====
    for traj, label in zip(trajs, labels):

        if label == "REF":
            ax.plot(
                traj[:, idx1],
                traj[:, idx2],
                color='black',
                linewidth=2.5,
                label=label
            )
        else:
            ax.plot(
                traj[:, idx1],
                traj[:, idx2],
                linewidth=1.5,
                label=label
            )

        # 起点
        ax.scatter(traj[0, idx1], traj[0, idx2], s=20)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# XY (North-East)
plot_projection(
    axes[0],
    all_trajs,
    all_labels,
    0, 1,
    "North (m)",
    "East (m)",
    "XY Projection"
)

# XZ (North-Up)
plot_projection(
    axes[1],
    all_trajs,
    all_labels,
    0, 2,
    "North (m)",
    "Up (m)",
    "XZ Projection"
)

# YZ (East-Up)
plot_projection(
    axes[2],
    all_trajs,
    all_labels,
    1, 2,
    "East (m)",
    "Up (m)",
    "YZ Projection"
)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=4, fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()