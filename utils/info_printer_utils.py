import torch
import psutil  # 用于获取系统资源使用情况


class InfoPrinter:
    def __init__(self):
        pass

    # 打印配置信息
    def print_config(self, config):
        print("Model: ", config.MODEL_NAME)
        print("Data: ", config.DATASET_NAME)
        print("Learning rate: ", config.LEARNING_RATE)
        print("Batch size: ", config.BATCH_SIZE)
        print("Epochs: ", config.EPOCHS)
        print("Input size: ", config.INPUT_SIZE)
        print("Output size: ", config.OUTPUT_SIZE)
        print("Device: ", config.DEVICE)
        print("Model save directory: ", config.MODEL_SAVE_DIR)
        print("Task type: ", config.TASK_TYPE)
        print("Test interval: ", config.TEST_INTERVAL)

    # 打印模型结构和参数数量
    def print_model(self, model):
        print(model)
        print(
            "Number of parameters: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    # 打印训练信息
    def print_training_info(self, epoch, epochs, train_loss):
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Training Loss: {train_loss:.4f}")

    # 打印测试信息
    def print_test_info(
        self, epoch, test_loss, test_metric=None, task_type="classification"
    ):
        if task_type == "classification":
            print(f"Test Accuracy: {test_metric * 100:.2f}%")
        else:
            print(f"Test Loss: {test_loss:.4f}")
        print(f"Logged results for epoch {epoch}.")

    # 打印模型保存信息
    def print_model_save_info(self, model_path, best=False):
        if best:
            print(f"Best model saved to {model_path}")
        else:
            print(f"Last model saved to {model_path}")
