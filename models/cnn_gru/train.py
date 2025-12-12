from datetime import datetime

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_process import *
from model import *
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, input_multi, output, batch_size = 32):
        """
        input_multi.shape: [50, 100, 9, 2400]
        output.shape: [50, 2400, 3]
        """
        # 重新组织数据结构，将每个样本单独处理
        self.samples = []
        for batch_idx in range(min(input_multi.shape[0], batch_size)):  # 遍历batch
            for seq_idx in range(input_multi.shape[3]):  # 遍历每个batch中的序列
                # 提取单个序列 [100, 9] 和对应的标签 [3]
                self.samples.append((
                    input_multi[batch_idx, :, :, seq_idx],
                    output[batch_idx, seq_idx, :]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloader(input_multi, output, batch_size=32, shuffle=True):
    """创建DataLoader"""
    dataset = CustomDataset(input_multi, output)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, save_to_path:str = ''):
    """训练模型的主函数"""
    model.train()
    model.to(device)
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')

    plt.ioff()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, (data, target) in progress_bar:
            # 将数据移至设备
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (batch_idx + 1):.4f}')
        avg_loss = total_loss / len(train_loader)
        train_loss.append(avg_loss)
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            torch.save(model.state_dict(), save_to_path + '/best_train.pth')
            print(f"✅✅✅ 保存最佳训练模型（val_loss = {best_train_loss:.4f}）")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Val Loss: {avg_val_loss:.4f}')
        val_loss.append(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_to_path + '/best_val.pth')
            print(f"✅ 保存最佳模型（val_loss = {best_val_loss:.4f}）")
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(train_loader):.4f}')

    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.show()

def run(file_path:str = '/Users/yangyu/PycharmProjects/gnss-ins-sim/cnn_gru/demo_saved_data/drone_sim'):
    pre_len = 10
    # input.multi.shape:[50, pre_len, 9, 2400]
    # output.shape: [50, 2400, 3]
    input_multi, output = get_input_output_data(pre_len, file_path=file_path)

    total_samples = input_multi.shape[0]
    indices = torch.randperm(total_samples)
    split_idx = int(total_samples * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    input_train, output_train = input_multi[train_idx], output[train_idx]
    input_val, output_val = input_multi[val_idx], output[val_idx]
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    save_to_path = f"val_info/{current_time}"
    os.makedirs(save_to_path, exist_ok=True)
    torch.save({
        'input_train': input_train,
        'output_train': output_train
    }, save_to_path + '/train_data.pth')
    torch.save({
        'input_train': input_val,
        'output_train': output_val
    }, save_to_path + '/val_data.pth')
    print("✅ 已保存训练集数据到 " + save_to_path + "/train_data.pth")

    config = argparse.Namespace(
        input_size=input_multi.shape[2],  # 9
        cnn_out=64,
        gru_hidden=128,
        output_size=output.shape[2],  # 3
        lr = 0.001
    )

    # 创建模型
    model = CNN_GRU(configs=config)

    # 创建数据加载器
    train_loader = create_dataloader(input_train, output_train, batch_size=32, shuffle=True)
    val_loader = create_dataloader(input_val, output_val, batch_size=32, shuffle=True)

    # 定义损失函数和优化器
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    # 设置设备
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("training on", device)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=1000, save_to_path=save_to_path)


if __name__ == '__main__':
    run(file_path='/sim_data_gen/sim_files/saved_file/constant_vertical_fall')
