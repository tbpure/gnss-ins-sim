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


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """训练模型的主函数"""
    model.train()
    model.to(device)
    train_loss = []
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
            torch.save(model.state_dict(), 'best_train.pth')
            print(f"✅✅✅ 保存最佳训练模型（val_loss = {best_train_loss:.4f}）")

        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(train_loader):.4f}')

    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.show()

def run():
    pre_len = 100
    # input.multi.shape:[50, 100, 9, 2400]
    # output.shape: [50, 2400, 3]
    input_multi, output = get_input_output_data(pre_len, file_path='/Users/bytedance/PycharmProjects/gnss-ins-sim/cnn_gru/demo_saved_data/drone_sim')

    os.makedirs("saved_data", exist_ok=True)
    torch.save({
        'input_train': input_multi,
        'output_train': output
    }, "saved_data/train_data.pth")
    print("✅ 已保存训练集数据到 saved_data/train_data.pth")

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
    train_loader = create_dataloader(input_multi, output, batch_size=32, shuffle=True)

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
    train_model(model, train_loader, criterion, optimizer, device, epochs=100)


if __name__ == '__main__':
    run()
