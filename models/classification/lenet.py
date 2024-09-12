import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LeNet, self).__init__()
        self.input_channels = input_size[0]  # 通道数
        self.output_size = output_size

        # 定义卷积层
        self.conv1 = nn.Conv2d(self.input_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # 通过input_size计算出展平后的大小
        self.fc_input_size = self._compute_fc_input_size(input_size)

        # 定义全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_size)

    def _compute_fc_input_size(self, input_size):
        # 模拟数据通过卷积层和池化层后的输出大小
        x = torch.randn(1, *input_size)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
