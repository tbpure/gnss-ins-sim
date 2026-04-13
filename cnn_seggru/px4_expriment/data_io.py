import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# IEEE 风格设置
# ======================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
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
})

# ======================
# 算法配色（自动匹配列名）
# ======================
algo_configs = {
    'REF': {
        'label': '真实轨迹 (Ground Truth)',
        'color': '#6E6E6E',
        'ls': '--',
        'lw': 2.0,
        'z': 20
    },
    'INS': {
        'label': 'IMU原始解',
        'color': '#B0B0B0',
        'ls': ':',
        'lw': 1.2,
        'z': 5
    },
    'CNN-SEGGRU': {
        'label': 'CNN-SEGGRU',
        'color': '#4C72B0',
        'ls': '-',
        'lw': 2.4,
        'z': 18
    },
    'GRU': {
        'label': 'GRU方法',
        'color': '#55A868',
        'ls': '--',
        'lw': 1.8,
        'z': 10
    },
    'CNN-GRU': {
        'label': 'CNN-GRU方法',
        'color': '#C44E52',
        'ls': '-.',
        'lw': 1.8,
        'z': 12
    }
}

directions = {
    "North": "北向 (North)",
    "East": "东向 (East)",
    "Height": "天向 (Up)"
}

file_path = "datas/data.xlsx"
sheets = ["North", "East", "Height"]

# ======================
# 读取 Excel
# ======================
dfs = pd.read_excel(file_path, sheet_name=sheets)

rmse_results = {d: {} for d in sheets}
data_cache = {}

# ======================
# 数据整理
# ======================
for d in sheets:
    df = dfs[d].copy()

    # 自动识别 ref
    if "REF" in df.columns:
        ref = df["REF"].values
    else:
        ref = df.iloc[:, 0].values
        df = df.iloc[:, 1:]

    data_cache[d] = {"REF": ref}

    for col in df.columns:
        data_cache[d][col] = df[col].values


# ======================
# 图1：轨迹对比
# ======================
fig1, axes1 = plt.subplots(3, 1, figsize=(10, 10))  # ⭐改这里

for idx, d in enumerate(sheets):
    ax = axes1[idx]

    for algo, cfg in algo_configs.items():
        if algo not in data_cache[d]:
            continue

        ax.plot(
            data_cache[d][algo],
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=cfg["lw"],
            zorder=cfg["z"]
        )

    ax.set_title(directions[d])
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("位置 (m)")
handles, labels = axes1[0].get_legend_handles_labels()

fig1.tight_layout(rect=[0, 0, 1, 0.92])

fig1.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),
    ncol=3   # ⭐3行图建议减少列数
)

# ======================
# 图2：误差 + RMSE
# ======================
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 10))  # ⭐改这里

for idx, d in enumerate(sheets):
    ax = axes2[idx]

    ref = data_cache[d]["REF"]

    for algo, cfg in algo_configs.items():
        if algo == "REF":
            continue
        if algo not in data_cache[d]:
            continue

        y = data_cache[d][algo]
        error = y - ref

        rmse = np.sqrt(np.mean(error ** 2))
        rmse_results[d][algo] = rmse

        ax.plot(
            error,
            label=f"{cfg['label']} (RMSE={rmse:.2f} m)",
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=cfg["lw"],
            zorder=cfg["z"]
        )

    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.set_title(directions[d])
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("误差 (m)")

handles2, labels2 = axes2[0].get_legend_handles_labels()

fig2.tight_layout(rect=[0, 0, 1, 0.92])


fig2.legend(
    handles2, labels2,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),
    ncol=2
)


# ======================
# 保存
# ======================
import os
os.makedirs("figures", exist_ok=True)

fig1.savefig("figures/trajectory_comparison.pdf", bbox_inches='tight')
fig1.savefig("figures/trajectory_comparison.png", dpi=600, bbox_inches='tight')
fig2.savefig("figures/error_comparison.pdf", bbox_inches='tight')
fig2.savefig("figures/error_comparison.png", dpi=600, bbox_inches='tight')

plt.show()


# ======================
# RMSE 表格
# ======================
rmse_table = pd.DataFrame(rmse_results).T

rmse_table = rmse_table.round(2)

print("\n===== RMSE Results (m) =====")
print(rmse_table)