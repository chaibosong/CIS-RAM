import torch
import torch.nn as nn


# 使用hnet提取雷达信号特征
class HNet(nn.Module):
    def __init__(self, num_classes=2):
        super(HNet, self).__init__()

        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, bias=False, padding=2),
            nn.BatchNorm1d(8, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, bias=False, padding=2),
            nn.BatchNorm1d(4, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2)
        )

        # self.lstm = nn.LSTM(
        #     input_size=4,
        #     # hidden_size=8,
        #     hidden_size=2,
        #     batch_first=True,
        #     bidirectional=False,
        # )

        self.gru = nn.GRU(
            input_size=32,
            # hidden_size=8,
            hidden_size=8,
            batch_first=True,
            bidirectional=True,
        )

        # hnet 分类器
        # self.fc = nn.Sequential(
        #     nn.Linear(64, self.num_classes, bias=False)
        # )

    def forward(self, x):

        x = x.unsqueeze(0)  # [batch,1,32]

        x = x.unsqueeze(0)  # [batch,1,32]

        x = self.conv(x)  # [batch,out,32]

        # LSTM 模块
        # x, (h_n, c_n) = self.lstm(x)  # [batch,input,hidden]

        # GRU 模块
        x, h_n = self.gru(x)

        x = x.contiguous().view(x.size(0), -1)  # [batch,input*hidden]

        # 分类器
        # y = self.fc(x)  # [batch,rep_dim]

        return x


if __name__ == '__main__':
    x = torch.rand(128)
    model = HNet(2)
    y = model(x)
    print(y.shape)