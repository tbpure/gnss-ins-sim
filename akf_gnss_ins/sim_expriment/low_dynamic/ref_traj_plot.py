import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
from utils.data_io import get_gnss_data_from_path

# ======================
# 1. 全局样式
# ======================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 2. 读取与预处理
# ======================
path = "/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2026-04-17-18-26-58"
pos = get_gnss_data_from_path(path, True)
pos -= pos[0, :]
dt = 0.01

vel = np.diff(pos, axis=0) / dt
speed = np.linalg.norm(vel, axis=1)
speed = np.insert(speed, 0, speed[0])
speed = gaussian_filter1d(speed, sigma=2)
norm = plt.Normalize(vmin=speed.min(), vmax=speed.max())

# ======================
# 3. 绘图执行
# ======================
# figsize 宽度设大一点，方便两个图并排
fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)


def plot_optimized(ax, data_indices, title, xlabel, ylabel):
    p_sub = pos[:, data_indices]

    # 计算该视图下两个轴的各自跨度
    d_h = p_sub[:, 0].max() - p_sub[:, 0].min()
    d_v = p_sub[:, 1].max() - p_sub[:, 1].min()

    # 取当前视图两个轴的最大跨度，而不是全局三个轴的最大跨度
    # 这确保了轨迹在当前画布中是“撑满”的，同时保持 1:1 比例
    local_range = max(d_h, d_v) * 1.1

    mid_h = (p_sub[:, 0].max() + p_sub[:, 0].min()) / 2
    mid_v = (p_sub[:, 1].max() + p_sub[:, 1].min()) / 2

    # 线段处理
    segments = np.concatenate([p_sub[:-1, np.newaxis, :], p_sub[1:, np.newaxis, :]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2.5, zorder=2)
    lc.set_array(speed)
    ax.add_collection(lc)

    # 绘制起终点
    ax.scatter(p_sub[0, 0], p_sub[0, 1], c='green', s=50, label='起点', zorder=5)
    ax.scatter(p_sub[-1, 0], p_sub[-1, 1], c='red', s=50, label='终点', zorder=5)

    # 设置局部最优范围
    ax.set_xlim(mid_h - local_range / 2, mid_h + local_range / 2)
    ax.set_ylim(mid_v - local_range / 2, mid_v + local_range / 2)

    # 关键设置
    ax.set_aspect('equal', adjustable='box')  # 保证 1:1 比例且外框自适应
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax.set_title(title, fontweight='bold', size=14, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return lc


# 执行绘制
lc_obj = plot_optimized(axes[0], [0, 1], "XY 平面轨迹 (水平面)", "X (m)", "Y (m)")
plot_optimized(axes[1], [0, 2], "XZ 平面轨迹 (纵剖面)", "X (m)", "Z (m)")

# 图例优化
axes[0].legend(loc='best')
# axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# 颜色条
cbar = fig.colorbar(lc_obj, ax=axes, shrink=0.6, aspect=30, pad=0.04)
cbar.set_label("速度 (m/s)", fontweight='bold')
plt.savefig("result/ref_traj_with_speed.png", dpi=600, bbox_inches='tight')
plt.show()