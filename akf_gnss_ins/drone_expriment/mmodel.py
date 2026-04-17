import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d

# 简单MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

def fake_nn_prediction_torch(x, y, epochs=200, low_freq_avg = 0.6, high_freq_avg = 0.2):
    # 转tensor
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 👉 故意“训练得不够好”
    for epoch in range(epochs):
        model.train()

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 预测
    model.eval()
    with torch.no_grad():
        y_pred = model(x).squeeze().numpy()

    # 👉 加“平滑噪声”（关键！）
    noise = np.random.randn(len(y_pred)) * low_freq_avg * np.std(y_pred)

    # 低频（平滑）
    smooth_noise = gaussian_filter1d(noise, sigma=2)

    # 高频（不平滑）
    high_freq_noise = np.random.randn(len(y_pred)) * high_freq_avg * np.std(y_pred)

    # ✅ 混合
    final_noise = 0.7 * smooth_noise + 0.3 * high_freq_noise

    return y_pred + final_noise

if __name__ == "__main__":
    from data_io import ecef
    y = ecef[:, 0]


    # ✅ 标准化
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / y_std

    x = np.arange(len(y))
    x = x / x.max()  # 也顺便归一化

    y_pred_norm = fake_nn_prediction_torch(x, y_norm, epochs=500)

    # ✅ 还原
    y_pred = y_pred_norm * y_std + y_mean
    import matplotlib.pyplot as plt

    plt.plot(x, y, label="True")
    plt.plot(x, y_pred, label="Fake NN Prediction")
    plt.legend()
    plt.show()
