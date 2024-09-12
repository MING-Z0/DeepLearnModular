import os
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir, model_name, dataset_name):
        self.log_dir = os.path.join(log_dir, f"{model_name}_{dataset_name}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_scalar(self, tag, value, step):
        """记录标量数据"""
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """关闭 SummaryWriter"""
        self.writer.close()
