import torch.nn as nn
from models.base_model import BaseModel


class SoftmaxRegression(BaseModel):
    def __init__(self, input_size, output_size):
        super(SoftmaxRegression, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = super().forward(x)  # 展平操作
        x = self.fc(x)
        return self.softmax(x)
