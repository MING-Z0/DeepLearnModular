from models.regression.linear_model import LinearModel
from models.classification.softmax_regression import SoftmaxRegression
from models.both.mlp import MLP
from models.classification.lenet import LeNet
from models.classification.nin import NiN
from models.classification.googlenet import GoogLeNet


def load_model(model_name, input_size, output_size, task_type="regression"):
    """根据模型名称动态加载模型"""
    if model_name == "LinearModel":
        return LinearModel(input_size=input_size, output_size=output_size)
    elif model_name == "SoftmaxRegression":
        return SoftmaxRegression(input_size=input_size, output_size=output_size)
    elif model_name == "MLP":
        return MLP(input_size=input_size, output_size=output_size, task_type=task_type)
    elif model_name == "LeNet":
        return LeNet(input_size=input_size, output_size=output_size)
    elif model_name == "NiN":
        return NiN(input_size=input_size, output_size=output_size)
    elif model_name == "GoogLeNet":
        return GoogLeNet(input_size=input_size, output_size=output_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
