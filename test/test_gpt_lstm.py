import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ================================
# 1. 生成模拟数据
# ================================
def generate_simulated_data(seq_len=200, num_samples=1000):
    """生成IMU加速度和角速度，以及对应的伪GNSS位置"""
    imu_data = []
    gnss_data = []
    dt = 0.1  # 时间步长
    for _ in range(num_samples):
        # 随机生成速度变化，积分得到位置
        vel = np.cumsum(np.random.randn(seq_len, 2) * 0.1, axis=0)  # x,y速度
        pos = np.cumsum(vel * dt, axis=0)  # 积分得到位置

        # 加速度 = 速度差分 / dt
        acc = np.diff(np.vstack([np.zeros((1, 2)), vel]), axis=0) / dt
        # 角速度模拟为随机小角度变化
        gyro = np.random.randn(seq_len, 2) * 0.01

        imu_sample = np.hstack([acc, gyro])
        imu_data.append(imu_sample)
        gnss_data.append(pos)

    imu_data = np.array(imu_data, dtype=np.float32)
    gnss_data = np.array(gnss_data, dtype=np.float32)
    return imu_data, gnss_data

# ================================
# 2. 定义Dataset
# ================================
class IMUGNSSDataset(Dataset):
    def __init__(self, imu_data, gnss_data):
        self.imu_data = imu_data
        self.gnss_data = gnss_data

    def __len__(self):
        return len(self.imu_data)

    def __getitem__(self, idx):
        return self.imu_data[idx], self.gnss_data[idx]

# ================================
# 3. 定义GRU/LSTM模型
# ================================
class IMU2GNSS(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=2, rnn_type='GRU'):
        super().__init__()
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'GRU' or 'LSTM'")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # rnn_out: [batch, seq_len, hidden]
        out = self.fc(rnn_out)     # 输出每个时间步的位置预测
        return out

# ================================
# 4. 训练函数
# ================================
def train_model(model, dataloader, num_epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in dataloader:
            x = x
            y = y
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}")

# ================================
# 5. 测试函数
# ================================
def test_model(model, imu_sample, gnss_sample):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(imu_sample[None, ...]))
    pred = pred.cpu().numpy()[0]
    plt.figure(figsize=(6,6))
    plt.plot(gnss_sample[:,0], gnss_sample[:,1], 'g-', label='True GNSS')
    plt.plot(pred[:,0], pred[:,1], 'r--', label='Predicted')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('IMU to GNSS Position Prediction')
    plt.show()

# ================================
# 6. 主程序
# ================================
if __name__ == "__main__":
    # 生成数据
    imu_data, gnss_data = generate_simulated_data(seq_len=200, num_samples=1000)
    dataset = IMUGNSSDataset(imu_data, gnss_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型
    model = IMU2GNSS(input_size=4, hidden_size=64, num_layers=2, output_size=2, rnn_type='GRU')

    # 训练
    train_model(model, dataloader, num_epochs=20, lr=1e-3)

    # 测试
    test_model(model, imu_data[0], gnss_data[0])