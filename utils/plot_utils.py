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