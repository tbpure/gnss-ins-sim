import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_segrnn import Model


# 数据准备
train_X = torch.randn(100, 10, 9)  # (samples, seq_len, channels)
train_Y = torch.randn(100, 5, 2)   # (samples, pred_len, 2)
dataset = TensorDataset(train_X, train_Y)
dataloader = DataLoader(dataset, batch_size=32)

# 模型初始化
configs = argparse.Namespace(
    seq_len=10,
    pred_len=5,
    enc_in=9,
    d_model=64,
    dropout=0.1,
    seg_len=5
)
model = Model(configs)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(100):
    for x, y in dataloader:
        pred = model(x)  # 输出形状：[batch, pred_len, 2]
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if  __name__ == '__main__':
    model.eval()
    with torch.no_grad():
        x, y = next(iter(dataloader))
        pred = model(x)
        print(pred.shape)