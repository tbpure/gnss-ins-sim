import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gnss_pre.dataset_gen import get_dataset_from_single_path


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=20, output_size=6):
        super(LSTMModel, self).__init__()

        # 第一层 LSTM，return_sequences=True → batch_first=True + output
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )

        # 第二层 LSTM，return_sequences=False → 只取 last hidden
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )

        # Dense(6)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        out, _ = self.lstm1(x)      # (batch, seq_len, 20)
        out, _ = self.lstm2(out)    # (batch, seq_len, 20)

        # return_sequences=False → 取最后一个 time step
        last_out = out[:, -1, :]    # (batch, 20)

        output = self.fc(last_out)  # (batch, 6)
        return output


class WeightedMAE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        # 确保是 tensor
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        # 自动 broadcast，假设 shape = (batch, dim)
        loss = torch.abs(y_pred - y_true) * self.weights.to(y_pred.device)
        return loss.mean()


class IMUGNSSDataset(Dataset):
    def __init__(self, file_list, step):
        self.features = []
        self.labels = []

        for fp in file_list:
            f, l = get_dataset_from_single_path(fp, step)
            self.features.append(f)
            self.labels.append(l)

        self.features = np.vstack(self.features)
        self.labels = np.vstack(self.labels)

        print("Dataset loaded.")
        print("Features shape:", self.features.shape)  # (N, step, imu_dim)
        print("Labels shape:", self.labels.shape)      # (N, gnss_dim)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def get_dataloader(file_list, step, batch_size=64, shuffle=True):
    dataset = IMUGNSSDataset(file_list, step)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(train_loader, input_dim, epochs=10, lr=1e-3, device="mps"):
    device = torch.device(device if torch.mps.is_available() else "cpu")
    print(f'traing model on {device}')
    model = LSTMModel(input_size=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    signals_weights_tensor = np.array([3.8, 3.9, 7.6, 1, 1, 5.5])
    criterion = WeightedMAE(signals_weights_tensor)


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss/count:.4f}")

    return model


if __name__ == "__main__":
    files = [
        "/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2025-11-10-20-00-21",
        "/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2025-11-08-20-41-11",
        "/Users/yangyu/PycharmProjects/gnss-ins-sim/sim_data_gen/sim_files/saved_file/motion_def-90deg_turn_long/2025-11-08-20-04-25"
    ]

    step = 10  # 时间序列长度
    batch_size = 2

    # 生成 DataLoader
    train_loader = get_dataloader(files, step, batch_size)

    # 获取 IMU 单帧输入维度
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]  # e.g., 6

    # 训练
    model = train_model(train_loader, input_dim, epochs=500, lr=1e-3)