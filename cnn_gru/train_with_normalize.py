import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_process import *
from model import *
import argparse
from tqdm import tqdm


# 数据归一化类（支持 PyTorch）
class Normalizer:
    def __init__(self, method='z-score'):
        self.method = method
        self.params = {}

    def fit(self, data: torch.Tensor):
        """data: shape (N, D)"""
        if self.method == 'min-max':
            self.params['min'] = data.min(dim=0).values
            self.params['max'] = data.max(dim=0).values
        elif self.method == 'z-score':
            self.params['mean'] = data.mean(dim=0)
            self.params['std'] = data.std(dim=0)
            self.params['std'][self.params['std'] < 1e-10] = 1.0
        elif self.method == 'robust':
            q1 = data.kthvalue(int(0.25 * data.size(0)), dim=0).values
            q3 = data.kthvalue(int(0.75 * data.size(0)), dim=0).values
            self.params['median'] = data.median(dim=0).values
            self.params['q1'] = q1
            self.params['q3'] = q3
            self.params['iqr'] = q3 - q1
            self.params['iqr'][self.params['iqr'] < 1e-10] = 1.0
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")
        return self

    def transform(self, data: torch.Tensor):
        if self.method == 'min-max':
            return (data - self.params['min']) / (self.params['max'] - self.params['min'] + 1e-10)
        elif self.method == 'z-score':
            return (data - self.params['mean']) / self.params['std']
        elif self.method == 'robust':
            return (data - self.params['median']) / self.params['iqr']
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

    def inverse_transform(self, data: torch.Tensor):
        if self.method == 'min-max':
            return data * (self.params['max'] - self.params['min'] + 1e-10) + self.params['min']
        elif self.method == 'z-score':
            return data * self.params['std'] + self.params['mean']
        elif self.method == 'robust':
            return data * self.params['iqr'] + self.params['median']
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")


# 自定义数据集类（延迟归一化）
class CustomDataset(Dataset):
    def __init__(self, input_multi, output, input_normalizer=None, output_normalizer=None):
        """
        input_multi.shape: [B, T, C, N]
        output.shape: [B, N, D]
        """
        self.input_multi = input_multi
        self.output = output
        self.total_samples = input_multi.shape[0] * input_multi.shape[3]

        # 归一化器
        if input_normalizer is None:
            input_flattened = input_multi.permute(0, 3, 1, 2).reshape(-1, input_multi.shape[1], input_multi.shape[2])
            input_flattened = input_flattened.reshape(-1, input_multi.shape[2])
            self.input_normalizer = Normalizer(method='z-score').fit(input_flattened)
        else:
            self.input_normalizer = input_normalizer

        if output_normalizer is None:
            output_flattened = output.reshape(-1, output.shape[2])
            self.output_normalizer = Normalizer(method='z-score').fit(output_flattened)
        else:
            self.output_normalizer = output_normalizer

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        batch_idx = idx // self.input_multi.shape[3]
        seq_idx = idx % self.input_multi.shape[3]

        # 提取单个样本
        x = self.input_multi[batch_idx, :, :, seq_idx]  # [T, C]
        y = self.output[batch_idx, seq_idx, :]         # [D]

        # 应用归一化
        x_norm = self.input_normalizer.transform(x)
        y_norm = self.output_normalizer.transform(y)

        return x_norm, y_norm


def create_dataloader(input_multi, output, batch_size=50, shuffle=True, num_workers=4, pin_memory=True):
    dataset = CustomDataset(input_multi, output)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(progress_bar))

        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.4f}')


def run():
    pre_len = 100
    input_multi, output = get_input_output_data(pre_len)

    config = argparse.Namespace(
        input_size=input_multi.shape[2],
        cnn_out=64,
        gru_hidden=128,
        output_size=output.shape[2]
    )

    model = CNN_GRU(config)
    train_loader = create_dataloader(input_multi, output, batch_size=50)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_model(model, train_loader, criterion, optimizer, device, epochs=10)

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_normalizer': train_loader.dataset.input_normalizer,
        'output_normalizer': train_loader.dataset.output_normalizer
    }, 'cnn_gru_model.pth')


if __name__ == "__main__":
    run()
