import torch.nn as nn
from models.base_model import BaseModel


class MLP(BaseModel):
    def __init__(self, input_size, output_size, task_type="classification"):
        super(MLP, self).__init__()
        self.hidden_size = 64  # 可以根据需求进行调整
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, output_size)
        self.task_type = task_type

    def forward(self, x):
        x = self.flatten(x)  # 展平输入
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # 根据任务类型调整输出
        if self.task_type == "regression":
            x = x.squeeze(-1)  # 回归任务 如果最后一个维度为 1，则删除
        elif self.task_type == "classification":
            x = nn.Softmax(dim=1)(x)  # 分类任务 应用Softmax

        return x
