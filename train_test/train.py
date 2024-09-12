import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    # 创建一个GradScaler实例
    scaler = GradScaler()

    # 使用tqdm显示进度条
    progress_bar = tqdm(
        enumerate(train_loader), total=len(train_loader), desc="Training"
    )

    for i, (batch_X, batch_y) in progress_bar:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # 使用autocast进行半精度计算
        with autocast():
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

        optimizer.zero_grad()

        # 使用scaler进行反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        # 每10个batch输出一次平均损失
        if (i + 1) % 10 == 0:
            avg_loss = epoch_loss / (i + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    avg_loss = epoch_loss / len(train_loader)
    print(f"平均损失: {avg_loss:.4f}")

    return avg_loss  # 返回这个epoch的平均损失
