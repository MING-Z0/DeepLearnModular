import os
import torch


class Config:
    # 数据集相关配置
    NUM_WORKERS = 8  # 数据加载的线程数
    DATA_DIR = "./data"
    # {
    # "FashionMNIST", 28x28 out: 10
    # "MNIST", in: 28x28 out: 10
    # "Iris", in: 4 out: 3
    # "CaliforniaHousing" in: 8 out: 1
    # "Wine" in: 13 out: 3
    # "Digits" in: 8x8 out: 10
    # }
    DATASET_NAME = "FashionMNIST"
    BATCH_SIZE = 64

    # 模型相关配置
    # {
    # "LinearModel"
    # "SoftmaxRegression"
    # "MLP"
    # "CNN"
    # "LeNet"
    # "NiN"
    # "GoogLeNet"
    # }
    MODEL_NAME = "GoogLeNet"
    INPUT_SIZE = (1, 28, 28)  # 输入图像的尺寸 (channels, height, width)
    OUTPUT_SIZE = 10  # 分类数或回归任务的输出维度

    # 任务类型
    TASK_TYPE = "classification"  #  "classification"(分类), "regression"(回归)

    # 训练相关配置
    LEARNING_RATE = 0.001  # 学习率
    NUM_WORKERS = 4  # 数据加载的线程数
    EPOCHS = 10
    TEST_INTERVAL = 2  # 每隔多少个epoch进行一次测试

    # 设备相关配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 日志和模型保存路径
    LOG_DIR = f"./logs/{MODEL_NAME}_lr={LEARNING_RATE}_batch_size={BATCH_SIZE}_epochs={EPOCHS}_dataset={DATASET_NAME}"
    MODEL_SAVE_DIR = f"./saved_models/{MODEL_NAME}/{DATASET_NAME}/"
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_best.pth")
    LAST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_last.pth")

    def __init__(self):
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

    # 获取损失函数
    def get_criterion(self):
        if self.TASK_TYPE == "classification":
            return torch.nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
        elif self.TASK_TYPE == "regression":
            return torch.nn.MSELoss()  # 回归任务使用均方误差损失
        else:
            raise ValueError(f"Unsupported task type: {self.TASK_TYPE}")

    # 获取优化器
    def get_optimizer(self, model_parameters):
        return torch.optim.Adam(model_parameters, lr=self.LEARNING_RATE)
