import os
import torch
import threading
import copy
from configs.config import Config
from dataset.datasets import get_dataloaders  # 引入数据加载函数
from train_test.train import train
from train_test.test import test
from models import load_model  # 引入模型加载函数
from utils.tensorboard_utils import TensorBoardLogger  # 引入TensorBoard日志模块
from utils.info_printer_utils import InfoPrinter  # 引入打印配置信息的类


def load_data(config):
    """加载数据并返回训练和测试数据加载器"""
    return get_dataloaders(
        config.DATASET_NAME,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR,
    )


def initialize_model(config):
    """初始化模型并移动到指定设备"""
    model = load_model(
        config.MODEL_NAME, config.INPUT_SIZE, config.OUTPUT_SIZE, config.TASK_TYPE
    )
    device = torch.device(config.DEVICE)
    model.to(device)
    return model, device


def setup_training_tools(model, config):
    """设置损失函数和优化器"""
    criterion = config.get_criterion()
    optimizer = config.get_optimizer(model.parameters())
    return criterion, optimizer


def setup_paths(config):
    """设置保存模型的路径"""
    model_save_dir = config.MODEL_SAVE_DIR
    os.makedirs(model_save_dir, exist_ok=True)
    paths = {
        "best_model_path": config.BEST_MODEL_PATH,
        "last_model_path": config.LAST_MODEL_PATH,
    }
    return paths


def async_save_model(model, path):
    """异步保存模型"""
    model_copy = copy.deepcopy(model)
    # 将模型转移到CPU上再进行保存，以减少GPU占用
    model_copy.cpu()

    def save():
        torch.save(model_copy.state_dict(), path)
        print(f"Model saved to {path}")

    thread = threading.Thread(target=save)
    thread.start()


def train_and_evaluate(model, train_loader, test_loader, config, paths, logger):
    """执行训练和测试，并保存模型"""
    best_metric = float("inf") if config.TASK_TYPE == "regression" else 0.0
    criterion, optimizer = setup_training_tools(model, config)

    print("开始训练...")
    for epoch in range(1, config.EPOCHS + 1):
        print(f"第 {epoch}/{config.EPOCHS} 轮训练")

        # 训练模型
        train_loss = train(model, train_loader, criterion, optimizer, config.DEVICE)
        logger.log_scalar("损失/训练", train_loss, epoch)  # 同步记录训练损失

        if epoch % config.TEST_INTERVAL == 0 or epoch == config.EPOCHS:
            test_loss = test(
                model, test_loader, criterion, config.DEVICE, config.TASK_TYPE
            )

            if config.TASK_TYPE == "classification":
                _, test_metric = test_loss
                print(f"测试准确率: {test_metric * 100:.2f}%")
                logger.log_scalar(
                    "准确率/测试", test_metric, epoch
                )  # 同步记录测试准确率

                # 保存最佳模型（分类任务中，保存最高准确率的模型）
                if test_metric > best_metric:
                    best_metric = test_metric
                    torch.save(model.state_dict(), paths["best_model_path"])
                    print(f"模型已保存到 {paths['best_model_path']}")

            else:
                test_metric = test_loss
                print(f"测试损失: {test_metric:.4f}")
                logger.log_scalar("损失/测试", test_metric, epoch)  # 同步记录测试损失

                # 保存最佳模型（回归任务中，保存最低损失的模型）
                if test_metric < best_metric:
                    best_metric = test_metric
                    torch.save(model.state_dict(), paths["best_model_path"])
                    print(f"模型已保存到 {paths['best_model_path']}")

    # 保存最后一次模型
    torch.save(model.state_dict(), paths["last_model_path"])
    print(f"最后模型已保存到 {paths['last_model_path']}")


def main():
    # 从配置文件中加载配置
    config = Config()

    # 初始化打印器
    printer = InfoPrinter()

    # 打印配置信息
    printer.print_config(config)

    # 加载数据
    train_loader, test_loader = load_data(config)

    # 初始化模型
    model, device = initialize_model(config)

    # 打印模型结构
    printer.print_model(model)

    # 设置路径
    paths = setup_paths(config)

    # 初始化TensorBoard记录器
    logger = TensorBoardLogger(
        log_dir=config.LOG_DIR,
        model_name=config.MODEL_NAME,
        dataset_name=config.DATASET_NAME,
    )

    # 执行训练和测试
    train_and_evaluate(model, train_loader, test_loader, config, paths, logger)

    # 关闭TensorBoard记录器
    logger.close()


if __name__ == "__main__":
    main()
