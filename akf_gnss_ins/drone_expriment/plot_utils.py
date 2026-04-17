import numpy as np
import matplotlib.pyplot as plt
colors = ['#E76F51', '#E9C46A', '#2A9D8F']
def reformat_by_direction(errors, labels):
    """
    将 errors 从 [algo][T,3] 转换为：
    {
        'N': [algo1_N, algo2_N, ...],
        'E': [...],
        'U': [...]
    }
    """

    N_list = []
    E_list = []
    U_list = []

    for error in errors:
        N_list.append(error[:, 0])
        E_list.append(error[:, 1])
        U_list.append(error[:, 2])

    return np.array(N_list), np.array(E_list), np.array(U_list), labels

def plot_bars(n, e, h, labels):
    x = np.arange(len(labels))
    width = 0.6

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- N方向 ---
    axes[0].bar(x, n, width)
    axes[0].set_title("North Direction Error")
    axes[0].set_ylabel("Error")

    # --- E方向 ---
    axes[1].bar(x, e, width)
    axes[1].set_title("East Direction Error")
    axes[1].set_ylabel("Error")

    # --- H方向 ---
    axes[2].bar(x, h, width)
    axes[2].set_title("Height Direction Error")
    axes[2].set_ylabel("Error")

    # x轴标签
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def plot_rmse_bars(ax, data, labels, colors=None, edge_color='black', bar_width=0.25):
    """
    绘制分组柱状图，用于对比不同算法在不同 Outage 下的 RMSE。

    参数:
    ax: matplotlib 的 axes 对象。
    data: 列表，每个元素是一个包含 N 个样本误差的 array/list (例如: [algo1_err, algo2_err, ...])。
    labels: 算法名称列表，用于图例 (例如: ['LSTM', 'GBDT', 'KF'])。
    colors: 颜色列表。
    edge_color: 柱子边缘颜色。
    bar_width: 单个柱子的宽度。
    """
    if colors is None:
        # 默认使用图中类似的配色方案
        colors = ['#E17A5D', '#E9C46A', '#2A9D8F', '#264653']

    num_algos = len(data)
    num_outages = len(data[0])

    # 设置 X 轴位置
    indices = np.arange(num_outages)

    # 计算起始偏移量，使柱状图群组居中
    total_group_width = num_algos * bar_width
    start_pos = indices - (total_group_width / 2) + (bar_width / 2)

    # 循环绘制每个算法的柱子
    for i in range(num_algos):
        ax.bar(
            start_pos + i * bar_width,
            data[i],
            width=bar_width,
            label=labels[i],
            color=colors[i % len(colors)],
            edgecolor=edge_color,
            linewidth=0.6,
            zorder=3  # 确保柱子在网格线上方
        )

    # 基础样式设置
    ax.set_xticks(indices)
    ax.set_xticklabels(range(1, num_outages + 1))  # 设置 Outage 编号从 1 开始

    # 细节美化
    ax.grid(True, axis='both', linestyle='-', alpha=0.3, zorder=0)
    ax.legend(loc='upper right', ncol=3, frameon=True, fontsize=10, edgecolor='black')

    # 限制坐标轴范围以匹配原图
    ax.set_xlim(-1, num_outages)


def plot_ape_by_datas(data:dict, ref:np.array, final_len = 60, save_path = None):
    # 这里dict value是一个n*3的np array, key是其label
    target_len = ref.shape[0]
    for value in data.values():
        target_len = min(target_len, len(value))
    ref = ref[::round(ref.shape[0] / target_len), :]
    # 三维数组
    errors = []
    labels = data.keys()
    for value in data.values():
        ratio = round(value.shape[0] / target_len)
        if ratio != 1:
            value = value[::ratio, :]
        error = value - ref
        errors.append(np.abs(error))
    labels = [x.upper() for x in labels]
    n, e, h, labels = reformat_by_direction(errors, labels)
    n = n[:, ::round(n.shape[1] / final_len)]
    e = e[:, ::round(e.shape[1] / final_len)]

    plot_configs = [
        {'data': n, 'ylabel': 'North APE/m'},
        {'data': e, 'ylabel': 'East APE/m'},
        # {'data': h, 'ylabel': 'Vertical APE/m'},
    ]

    fig, axes = plt.subplots(len(plot_configs), 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.2)  # 调整子图间距

    # 3. 循环绘制三个子图
    for i, config in enumerate(plot_configs):
        ax = axes[i]

        plot_rmse_bars(
            ax=ax,
            data=config['data'],
            labels=labels
        )

        # 设置细节
        ax.set_ylabel(config['ylabel'], fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='-', alpha=0.3)
        if (config['ylabel'] != "Vertical APE/m"):
            ax.set_ylim(0, 50)  # 根据原图设置 Y 轴刻度范围

        # 设置 X 轴刻度 (1-30)
        ax.set_xticks(range(len(config['data'][0])))
        ax.set_xticklabels(range(1, len(config['data'][0]) + 1))

    # 4. 设置最下方的 X 轴标签
    axes[-1].set_xlabel('APE', fontsize=12, fontweight='bold')

    # 5. 保存或展示
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    plt.show()
