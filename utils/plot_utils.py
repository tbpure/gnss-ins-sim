from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_3d_trajectory(ecef_data, title="ECEF 三维轨迹"):
    """
    绘制三维轨迹图
    :param ecef_data: ECEF坐标数组，shape为(N, 3)，列对应X、Y、Z
    :param title: 图表标题
    """
    # 数据校验
    if ecef_data.shape[1] != 3:
        raise ValueError(f"ECEF数据应为N×3的数组，当前形状为{ecef_data.shape}")

    # 提取X、Y、Z分量
    x = ecef_data[:, 0]
    y = ecef_data[:, 1]
    z = ecef_data[:, 2]

    # 创建3D画布
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹线
    ax.plot(x, y, z, linewidth=1.5, color='#2E86AB', label='ECEF轨迹')
    # 标记起点和终点
    ax.scatter(x[0], y[0], z[0], color='red', s=50, label='起点')
    ax.scatter(x[-1], y[-1], z[-1], color='green', s=50, label='终点')

    # 设置坐标轴标签
    ax.set_xlabel('ECEF-X (m)', fontsize=12)
    ax.set_ylabel('ECEF-Y (m)', fontsize=12)
    ax.set_zlabel('ECEF-Z (m)', fontsize=12)

    # 设置标题和图例
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='best', fontsize=10)

    # 优化显示
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_neu_and_3d(trajectories, labels, ref_idx=None):
    """
    trajectories: list of np.ndarray, each shape = (T, 3) -> [N, E, U]
    labels: list of str
    ref_idx: int or None, index of reference trajectory
    """

    plt.rcParams['font.size'] = 12

    # ======================
    # 1. NEU 分量对比图
    # ======================
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    directions = ['North (m)', 'East (m)', 'Up (m)']

    for i in range(3):
        for traj, label in zip(trajectories, labels):
            axs[i].plot(traj[:, i], label=label, linewidth=1.5)

        axs[i].set_ylabel(directions[i])
        axs[i].grid(True)

    axs[-1].set_xlabel('Time Step')
    axs[0].legend()
    fig.suptitle('NEU Position Comparison')

    # ======================
    # 2. 2D 平面轨迹（NE）
    # ======================
    plt.figure(figsize=(6, 6))
    for traj, label in zip(trajectories, labels):
        plt.plot(traj[:, 1], traj[:, 0], label=label)  # E vs N

    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('2D Trajectory (Top View)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    # ======================
    # 3. 3D 轨迹
    # ======================
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for traj, label in zip(trajectories, labels):
        ax.plot(traj[:, 1], traj[:, 0], traj[:, 2], label=label)  # E, N, U

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('3D Trajectory Comparison')

    ax.legend()

    plt.show()


# ======================
# 示例（你替换成自己的数据）
# ======================
if __name__ == "__main__":
    T = 500

    # 模拟数据
    t = np.linspace(0, 10, T)

    ref = np.stack([
        50 * np.sin(t),
        50 * np.cos(t),
        5 * t
    ], axis=1)

    algo1 = ref + np.random.normal(0, 1, ref.shape)
    algo2 = ref + np.random.normal(0, 2, ref.shape)

    trajectories = [ref, algo1, algo2]
    labels = ['Reference', 'EKF', 'Your Method']

    plot_neu_and_3d(trajectories, labels)