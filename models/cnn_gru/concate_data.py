import torch
import os
from matplotlib import pyplot as plt

from models.cnn_gru.model import CNN_GRU, config

def concate_data(file_path, model_name: str, x_index, y_index, z_index):

    if model_name == '':
        from pathlib import Path
        pwd = os.getcwd()
        model_files = list(Path(file_path + '/models').rglob('*.pth'))
        for model_path in model_files:
            concate_data(file_path, str(model_path.name), x_index, y_index, z_index)
        return

    label = ['x', 'y', 'z']
    index = [x_index, y_index, z_index]
    data = torch.load(file_path + "/train_data.pth")
    model_info = torch.load(file_path + "/models/" + model_name)
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
    real_data = real_data.to(device)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for i in range(3):
        input = input_data[index[i]].permute(2, 0, 1)
        model.eval()
        pre = model(input)
        real = real_data[index[i]].squeeze()

        pre = pre.detach().cpu().numpy()
        real = real.detach().cpu().numpy()

        # 将位置增量转换为真实位置轨迹
        pre_trajectory = pre.cumsum(axis=0)
        real_trajectory = real.cumsum(axis=0)
        axes[i].plot(pre_trajectory[:, i], label=f'Predicted Position', color='orange')
        axes[i].plot(real_trajectory[:, i], label=f'Real v', color='C0')
        axes[i].set_title(f'{label[i]}_{file_path}_{model_name}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Relative Position')
        axes[i].legend()

    print(f"🔍 加载模型: {model_name}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # x轴使用index为11的数据
    # y轴使用index为2的数据
    # z轴使用index为18的前80s数据
    concate_data("val_info/20250603_0946", "", 11, 2, 18)
