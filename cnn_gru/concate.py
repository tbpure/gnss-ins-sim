import torch
import os
from collections import defaultdict
from matplotlib import pyplot as plt

from cnn_gru.model import CNN_GRU, config

def concate_data(file_path, model_names: list, x_index, y_index, z_index):

    if len(model_names) == 0:
        from pathlib import Path
        model_files = list(Path(file_path + '/models').rglob('*.pth'))
        model_names = list(map(
            lambda x: str(x.relative_to(Path(file_path) / 'models')).replace('\\', '/'),
            model_files
        ))
        if len(model_names) != 0:
            concate_data(file_path, model_names, x_index, y_index, z_index)
        return

    label = ['x', 'y', 'z']
    index = [x_index, y_index, z_index]
    data = torch.load(file_path + "/train_data.pth")
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model_groups = defaultdict(list)
    for model_name in model_names:

        dir_name = os.path.dirname(model_name)
        if not dir_name:
            dir_name = "root"
        model_groups[dir_name].append(model_name)
    grouped_models = list(model_groups.values())
    for model_list in grouped_models:
        if len(model_list) > 10:
            sub_model_lists = [model_list[i:i + 10] for i in range(0, len(model_list), 10)]
        else:
            sub_model_lists = [model_list]
        for sub_model_list in sub_model_lists:
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            j = 0
            for model_name in sub_model_list:
                model_info = torch.load(file_path + "/models/" + model_name)
                model = CNN_GRU(configs=config)
                model.load_state_dict(model_info)
                model.to(device)

                input_data = data['input_train']
                real_data = data['output_train']
                input_data = input_data.to(device)
                real_data = real_data.to(device)
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
                    axes[i].plot(pre_trajectory[:, i], label=f'{model_name}')
                    if j == 0:
                        axes[i].plot(real_trajectory[:, i], label=f'Real Position')
                        dir_name = os.path.dirname(model_name) or "root"
                        axes[i].set_title(f'Model Group: {dir_name}')
                        axes[i].set_xlabel('Time Step')
                        axes[i].set_ylabel('Relative Position')
                    axes[i].legend()
                j += 1
                print(f"🔍 加载模型: {model_name}")
            plt.tight_layout()
            plt.show()



if __name__ == '__main__':
    # x轴使用index为11的数据
    # y轴使用index为2的数据
    # z轴使用index为18的前80s数据
    # for i in range(19):
        # concate_data("val_info/20250603_0946", [], i, i, i)
        concate_data("val_info/20250603_0946", [], 11, 2, 18)