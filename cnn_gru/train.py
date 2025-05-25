import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_process import *
from model import *
import argparse
from tqdm import tqdm


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, input_multi, output):
        """
        input_multi.shape: [50, 100, 9, 2400]
        output.shape: [50, 2400, 3]
        """
        # 重新组织数据结构，将每个样本单独处理
        self.samples = []
        for batch_idx in range(input_multi.shape[0]):  # 遍历batch
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


def create_dataloader(input_multi, output, batch_size=50, shuffle=True):
    """创建DataLoader"""
    dataset = CustomDataset(input_multi, output)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """训练模型的主函数"""
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, (data, target) in progress_bar:
            # 将数据移至设备
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)

            # 计算损失
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (batch_idx + 1):.4f}')

        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(train_loader):.4f}')


def run():
    pre_len = 100
    # input.multi.shape:[50, 100, 9, 2400]
    # output.shape: [50, 2400, 3]
    input_multi, output = get_input_output_data(pre_len)

    config = argparse.Namespace(
        input_size=input_multi.shape[2],  # 9
        cnn_out=64,
        gru_hidden=128,
        output_size=output.shape[2]  # 3
    )

    # 创建模型
    model = CNN_GRU(config)

    # 创建数据加载器
    train_loader = create_dataloader(input_multi, output, batch_size=50, shuffle=False)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, epochs=10)


if __name__ == '__main__':
    run()
