import torch
import torch.nn as nn


class NiN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NiN, self).__init__()
        self.input_channels = input_size[0]  # 通道数
        self.output_size = output_size

        # 定义NiN网络的卷积层，使用 1x1 卷积代替全连接层
        self.conv1 = nn.Conv2d(self.input_channels, 192, kernel_size=5, padding=2)
        self.nin1 = nn.Conv2d(192, 160, kernel_size=1)
        self.nin2 = nn.Conv2d(160, 96, kernel_size=1)

        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2)
        self.nin3 = nn.Conv2d(192, 192, kernel_size=1)
        self.nin4 = nn.Conv2d(192, 192, kernel_size=1)

        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.nin5 = nn.Conv2d(192, 192, kernel_size=1)
        self.nin6 = nn.Conv2d(192, self.output_size, kernel_size=1)

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一层卷积和NiN操作
        x = self.relu(self.nin1(self.relu(self.conv1(x))))
        x = self.relu(self.nin2(x))
        x = self.pool(x)

        # 第二层卷积和NiN操作
        x = self.relu(self.nin3(self.relu(self.conv2(x))))
        x = self.relu(self.nin4(x))
        x = self.pool(x)

        # 第三层卷积和NiN操作
        x = self.relu(self.nin5(self.relu(self.conv3(x))))
        x = self.nin6(x)

        # 全局平均池化层
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        return x
