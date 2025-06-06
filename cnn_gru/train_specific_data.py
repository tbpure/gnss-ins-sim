from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
from tqdm import tqdm

from cnn_gru.train import create_dataloader, train_model
from model import config, CNN_GRU
import torch
from torch import nn, optim


def train_specific_data(file_path: str, epochs: int = 10):
    data = torch.load(file_path + "/train_data.pth")
    val_data = torch.load(file_path + "/val_data.pth")
    # 创建模型
    model = CNN_GRU(configs=config)

    # 创建数据加载器
    train_loader = create_dataloader(data['input_train'], data['output_train'], batch_size=32, shuffle=False)
    val_loader = create_dataloader(val_data['input_train'], val_data['output_train'], batch_size=32, shuffle=False)

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
    print("💻training on device:", device)
    model.to(device)
    model.train()
    current_time = datetime.now().strftime("%m%d_%H%M")

    new_dir = Path(file_path) / 'models' / f'{current_time}_{epochs}'
    new_dir.mkdir(parents=True, exist_ok=True)
    save_model_to_dir = file_path + '/models/' + new_dir.name
    best_train_loss = float('inf')
    best_val_loss = float('inf')
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

            copy_output = output.detach().cpu().numpy().cumsum(axis=0)
            copy_target = target.detach().cpu().numpy().cumsum(axis=0)
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            for i in range(3):
                pass

            total_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (batch_idx + 1):.4f}')
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            if epoch < 50:
                torch.save(model.state_dict(), save_model_to_dir + '/best_train.pth')
            else:
                torch.save(model.state_dict(), save_model_to_dir + f'/best_train_{epoch}.pth')
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

        # 每N个epoch保存一次
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"{save_model_to_dir}/epoch_{epoch}.pth")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_model_to_dir + '/best_val.pth')
            print(f"✅ 保存最佳模型（val_loss = {best_val_loss:.4f}）")


if __name__ == '__main__':
    train_specific_data(file_path='val_info/20250603_0946', epochs=10000)
