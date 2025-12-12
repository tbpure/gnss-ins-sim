import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.output_len = configs.output_len

        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # 添加CNN层
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU()
        )

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1,
                          bias=True, batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len),
            nn.ReLU()
        )
        self.reshape = nn.Sequential(self.enc_in, self.output_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # 归一化并调整维度 [b, s, c] -> [b, c, s]
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)  # 维度变为 [b, c, s]

        # 新增：通过CNN处理时间序列
        x = self.cnn(x)  # 输出保持 [b, c, s]

        # 分块和嵌入
        x = self.valueEmbedding(x.view(-1, self.seg_num_x, self.seg_len))

        # 编码
        _, hn = self.rnn(x)

        # 位置和通道嵌入
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        # 解码
        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))

        # 预测
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_last  # 反调整维度并恢复归一化

        return y
