import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
from get_ref import north, east, high

# ======================
# 1. 全局样式
# ======================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
north /= 20
east /= 20
# ======================
# 2. 读取与预处理
# ======================
path = "/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2026-04-17-18-26-58"
pos = np.column_stack((north, east, high))
pos -= pos[0, :]
dt = 1

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


def plot_optimized(ax, data_indices, title, xlabel, ylabel, equal=True):
    p_sub = pos[:, data_indices]

    d_h = p_sub[:, 0].max() - p_sub[:, 0].min()
    d_v = p_sub[:, 1].max() - p_sub[:, 1].min()

    local_range = max(d_h, d_v) * 1.1

    mid_h = (p_sub[:, 0].max() + p_sub[:, 0].min()) / 2
    mid_v = (p_sub[:, 1].max() + p_sub[:, 1].min()) / 2

    segments = np.concatenate([p_sub[:-1, np.newaxis, :], p_sub[1:, np.newaxis, :]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2.5, zorder=2)
    lc.set_array(speed)
    ax.add_collection(lc)

    ax.scatter(p_sub[0, 0], p_sub[0, 1], c='green', s=50, label='起点', zorder=5)
    ax.scatter(p_sub[-1, 0], p_sub[-1, 1], c='red', s=50, label='终点', zorder=5)

    if equal:
        ax.set_xlim(mid_h - local_range / 2, mid_h + local_range / 2)
        ax.set_ylim(mid_v - local_range / 2, mid_v + local_range / 2)
        ax.set_aspect('equal', adjustable='box')
    else:
        # 自适应（关键）
        margin = 0.05
        ax.set_xlim(p_sub[:, 0].min() - margin*d_h, p_sub[:, 0].max() + margin*d_h)
        ax.set_ylim(p_sub[:, 1].min() - margin*d_v, p_sub[:, 1].max() + margin*d_v)

    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax.set_title(title, fontweight='bold', size=14, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return lc


# 执行绘制
lc_obj = plot_optimized(axes[0], [0, 1], "XY 平面轨迹", "X (m)", "Y (m)", equal=True)
plot_optimized(axes[1], [0, 2], "XZ 平面轨迹", "X (m)", "Z (m)", equal=False)
# 图例优化
axes[0].legend(loc='best')
# axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# 颜色条
cbar = fig.colorbar(lc_obj, ax=axes, shrink=0.6, aspect=30, pad=0.04)
cbar.set_label("速度 (m/s)", fontweight='bold')
plt.savefig("result/ref_traj_with_speed.png", dpi=600, bbox_inches='tight')
plt.show()



# ======================
# 三维轨迹（美化版）
# ======================
fig_3d = plt.figure(figsize=(8, 6))
ax3d = fig_3d.add_subplot(111, projection='3d')

x = pos[:, 0]
y = pos[:, 1]
z = pos[:, 2]

# ===== 1. 主轨迹（加粗 + 高级配色）
ax3d.plot(x, y, z, linewidth=2.5, color='#2A9D8F', label='参考运动轨迹')

# ===== 2. 起点终点（更醒目）
ax3d.scatter(x[0], y[0], z[0], c='#2ECC71', s=60, depthshade=True, label='起点')
ax3d.scatter(x[-1], y[-1], z[-1], c='#E74C3C', s=60, depthshade=True, label='终点')

# ===== 3. 坐标轴标签
ax3d.set_xlabel('北向 (m)', labelpad=10)
ax3d.set_ylabel('东向 (m)', labelpad=10)
ax3d.set_zlabel('天向 (m)', labelpad=10)

# ===== 4. 比例统一（防止形变）
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()

mid_x = (x.max()+x.min()) / 2
mid_y = (y.max()+y.min()) / 2
mid_z = (z.max()+z.min()) / 2

ax3d.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax3d.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax3d.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

# ===== 5. 视角优化（更有空间感）
ax3d.view_init(elev=25, azim=-60)

# ===== 6. 网格优化（更淡更高级）
ax3d.grid(True, linestyle='--', alpha=0.3)

# ===== 7. 去除背景面（关键高级感🔥）
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False

# 去掉边框线（更干净）
ax3d.xaxis.pane.set_edgecolor('w')
ax3d.yaxis.pane.set_edgecolor('w')
ax3d.zaxis.pane.set_edgecolor('w')

# ===== 8. 轴线变细（更精致）
ax3d.xaxis._axinfo["grid"]['linewidth'] = 0.5
ax3d.yaxis._axinfo["grid"]['linewidth'] = 0.5
ax3d.zaxis._axinfo["grid"]['linewidth'] = 0.5

# ===== 9. 图例
ax3d.legend(frameon=False, loc='upper left')

# ===== 10. 标题
plt.title("探空仪下落轨迹三维图", fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig("result/ref_traj_3d.png", dpi=600)
plt.show()