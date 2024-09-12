import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义残差模块（Basic Block）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # 第一层卷积，后跟批量归一化和 ReLU 激活函数
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二层卷积，后跟批量归一化
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 用于匹配维度的下采样操作

    def forward(self, x):
        identity = x  # 保留输入 x
        # 前向传播经过两层卷积
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 如果需要进行下采样（改变输入的维度），则进行下采样操作
        if self.downsample is not None:
            identity = self.downsample(x)

        # 将输入与输出相加（跳跃连接）
        out += identity
        out = F.relu(out)
        return out


# 定义 ResNet 主结构
class ResNet(nn.Module):
    def __init__(self, block, layers, input_size, output_size):
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始通道数

        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            input_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差网络层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_size)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # 如果输入和输出的通道数或尺寸不匹配，则进行下采样
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        # 创建多个残差块
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )  # 第一个残差块
        self.in_channels = out_channels  # 更新通道数

        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 创建 ResNet18 网络
def ResNet18(input_size, output_size):
    return ResNet(ResidualBlock, [2, 2, 2, 2], input_size, output_size)


# 创建 ResNet34 网络
def ResNet34(input_size, output_size):
    return ResNet(ResidualBlock, [3, 4, 6, 3], input_size, output_size)
