import argparse

import torch
import torch.nn as nn


class CNN_GRU(nn.Module):
    def __init__(self, configs):
        """
        config:
            input_size: 输入特征维度 (默认IMU的4个特征)
            cnn_out: CNN输出通道数
            gru_hidden: GRU隐藏层维度
            output_size: 输出维度
        """
        super(CNN_GRU, self).__init__()

        # 1D卷积层 (kernel_size=2, stride=1)
        self.input_size = configs.input_size
        self.cnn_out = configs.cnn_out
        self.gru_hidden = configs.gru_hidden
        self.output_size = configs.output_size
        self.conv = nn.Conv1d(
            in_channels=self.input_size,  # 输入特征数作为通道
            out_channels=self.cnn_out,  # 输出通道数
            kernel_size=2,
            stride=1
        )

        self.bn_after_conv = nn.BatchNorm1d(self.cnn_out)

        # 最大池化层 (kernel_size=3)
        self.pool = nn.MaxPool1d(
            kernel_size=3,
            stride=3  # 默认与kernel_size相同
        )

        # GRU层 (输入维度=CNN输出通道数)
        self.gru = nn.GRU(
            input_size=self.cnn_out,
            hidden_size=self.gru_hidden,
            batch_first=True  # 输入形状为(batch, seq, feature)
        )

        self.ln_after_gru = nn.LayerNorm(self.gru_hidden)

        self.ln_before_fc = nn.LayerNorm(self.gru_hidden)

        # 全连接输出层
        self.fc = nn.Linear(self.gru_hidden, self.output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播
        Input shape: (batch_size, seq_len, input_size)
        Output shape: (batch_size, output_size)
        """
        # 维度调整: [B, T, C] -> [B, C, T] (PyTorch卷积需要通道在前)
        x = x.permute(0, 2, 1)

        # 1D卷积 + ReLU
        x = self.conv(x)  # 输出形状: [B, cnn_out, T-1]
        x = self.bn_after_conv(x)
        x = self.relu(x)

        # 最大池化
        x = self.pool(x)  # 输出形状: [B, cnn_out, (T-1)//3]

        # 恢复维度: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)

        # GRU处理序列
        gru_out, _ = self.gru(x)  # gru_out形状: [B, seq_len, gru_hidden]
        gru_out = self.ln_after_gru(gru_out)
        # 取最后一个时间步的输出
        last_out = gru_out[:, -1, :]
        last_out = self.ln_before_fc(last_out)
        # 全连接层输出
        output = self.fc(last_out)

        return output


# 示例用法
if __name__ == "__main__":
    test_input_size = 9
    test_output_size = 3

    # 假设输入数据: batch_size=32, 序列长度100, 4个特征
    input_data = torch.randn(32, 100, test_input_size)

    config = argparse.Namespace(
        input_size=test_input_size,
        cnn_out=64,
        gru_hidden=128,
        output_size=test_output_size,
        lr = 0.01
    )
    # 初始化模型
    model = CNN_GRU(config)

    # 前向传播
    output = model(input_data)
    print("输出形状:", output.shape)  # 应该为 torch.Size([32, 3])
