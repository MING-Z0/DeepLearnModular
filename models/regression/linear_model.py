import torch.nn as nn
from models.base_model import BaseModel


class LinearModel(BaseModel):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)  # 展平操作
        x = self.fc(x)
        if x.shape[1] == 1:  # 如果输出是 [batch_size, 1]，则去除多余的维度
            x = x.squeeze(-1)
        return x
