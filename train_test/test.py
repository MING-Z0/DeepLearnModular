import torch
from tqdm import tqdm


def test(model, test_loader, criterion, device, task_type="classification"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")

    with torch.no_grad():
        for i, (batch_X, batch_y) in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            # 计算损失
            if task_type == "classification":
                loss = criterion(outputs, batch_y)

                # 获取预测结果
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            elif task_type == "regression":
                loss = criterion(outputs.squeeze(), batch_y)

            total_loss += loss.item()

            # 更新进度条上的损失值
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / (i + 1)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    avg_loss = total_loss / len(test_loader)

    # 如果是分类任务，计算准确率
    if task_type == "classification":
        accuracy = correct / total
        return avg_loss, accuracy
    else:
        # 对于回归任务，只返回平均损失
        return avg_loss
