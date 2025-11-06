import argparse
import os.path

import torch
from matplotlib import pyplot as plt

from cnn_gru.model import CNN_GRU


def load_saved_train_data(file_path:str, train_model:bool=True, train_data:bool = True):
    model_name = "best_train.pth" if train_model else "best_val.pth"
    load_specific_train_data(file_path, model_name, train_data)


def load_specific_train_data(file_path:str, model_name: str = '', train_data: bool = True):
    if model_name == '':
        from pathlib import Path
        model_files = list(Path(file_path + '/models').glob('*.pth'))
        for model_path in model_files:
            print(f"🔍 加载模型: {model_path}")
            load_specific_train_data(file_path, model_name=str(model_path), train_data=True)
        return

    data = torch.load(file_path + ("/train_data.pth" if train_data else "/val_data.pth"))
    if os.path.exists(file_path + '/models'):
        model_info = torch.load(file_path + '/models/'+ model_name)
    else:
        model_info = torch.load(file_path + '/'+ model_name)
    config = argparse.Namespace(
        input_size=9,
        cnn_out=64,
        gru_hidden=128,
        output_size=3,
        lr=0.001
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
    for i in range(real_data.shape[0]):
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
            axes[j].plot(pre_trajectory[:, j], label=f'Predicted Dimension {j + 1}', color='orange')
            axes[j].plot(real_trajectory[:, j], label=f'Real Dimension {j + 1}', color='C0')
            axes[j].set_title(f'Dimension {j + 1} Comparison')
            axes[j].set_xlabel('Time Step')
            axes[j].set_ylabel('Position')
            axes[j].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    load_specific_train_data("val_info/20250603_0946", "best_train.pth", train_data=False)
