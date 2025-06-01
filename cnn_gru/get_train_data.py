import argparse

import torch
from matplotlib import pyplot as plt

from cnn_gru.model import CNN_GRU


def load_saved_train_data(file_path:str):
    data = torch.load(file_path + "/train_data.pth")
    model_info = torch.load(file_path + "/best_train.pth")
    input_train = data['input_train']
    output_train = data['output_train']
    print(f"输入形状: {input_train.shape}, 标签形状: {output_train.shape}")

    config = argparse.Namespace(
        input_size=9,
        cnn_out=64,
        gru_hidden=128,
        output_size=3,
        lr = 0.001
    )
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model = CNN_GRU(configs=config)
    model.load_state_dict(model_info)
    model.to(device)

    input_data = data['input_train']
    real_data = data['output_train']
    input_data = input_data.to(device)
    for i in range(10):
        real_data = real_data.to(device)
        real = real_data[i].squeeze()
        input = input_data[i].permute(2, 0, 1)
        model.eval()
        pre = model(input)
        # 将张量移到 CPU 并转换为 NumPy 数组
        pre = pre.detach().cpu().numpy()
        real = real.detach().cpu().numpy()

        # 将位置增量转换为真实位置轨迹
        pre_trajectory = pre.cumsum(axis=0)
        real_trajectory = real.cumsum(axis=0)
        # 创建一个包含 3 个子图的画布
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        for j in range(3):
            axes[j].plot(pre_trajectory[:, j], label=f'Predicted Dimension {j + 1}')
            axes[j].plot(real_trajectory[:, j], label=f'Real Dimension {j + 1}')
            axes[j].set_title(f'Dimension {j + 1} Comparison')
            axes[j].set_xlabel('Time Step')
            axes[j].set_ylabel('Position')
            axes[j].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    load_saved_train_data("train_info/20250601_1805")
