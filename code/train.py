import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import save_checkpoint, plot_confusion_matrix, save_training_report
from config import config
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
import time


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑损失函数"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds, target):
        # 添加数值稳定性保护
        stable_preds = preds.clamp(min=-20.0, max=20.0)
        log_preds = torch.log_softmax(stable_preds, dim=-1)
        nll_loss = -log_preds.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_preds.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def train_model(model, train_loader, val_loader, device):
    # 选择损失函数 - 带标签平滑
    if config.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    else:
        # 添加稳定性保护的交叉熵损失
        criterion = nn.CrossEntropyLoss()

    # AdamW优化器添加权重衰减
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    if config.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, config.epochs, eta_min=config.min_lr)
    elif config.lr_schedule == 'step':
        scheduler = StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    else:
        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # 混合精度训练
    scaler = GradScaler(enabled=config.use_amp)

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"⚙️  训练配置: AMP={config.use_amp}, 标签平滑={config.label_smoothing}, 权重衰减={config.weight_decay}")
    print(f"⚙️  学习率调度: {config.lr_schedule}, epochs={config.epochs}")

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            if images.nelement() == 0 or labels.nelement() == 0:
                continue

            images, labels = images.to(device), labels.to(device)

            # 混合精度训练
            with autocast(enabled=config.use_amp):
                optimizer.zero_grad()
                outputs = model(images)

                # 添加输出稳定性保护
                outputs = outputs.clamp(min=-20.0, max=20.0)

                loss = criterion(outputs, labels)

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            if config.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

            scaler.step(optimizer)
            scaler.update()

            # 统计信息
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        # 计算训练指标
        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = 100. * correct / total

        # 验证阶段
        val_loss, val_acc, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

        # 更新学习率调度器
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 打印训练进度
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{config.epochs}] | "
              f"Time: {epoch_time:.0f}s | LR: {lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc)

            # 保存混淆矩阵
            if config.plot_confusion_matrix and config.confusion_matrix_size and len(all_preds) > 0:
                cm = plot_confusion_matrix(all_labels, all_preds,
                                           title=f'Best Confusion Matrix @ Epoch {epoch + 1}',
                                           figsize=config.confusion_matrix_size)
                report = save_training_report(all_labels, all_preds, val_acc, config.classes)

    # 训练结束
    print(f"\n🏆 训练完成! 最佳验证准确率: {best_acc:.2f}%")
    print(f"✅ 完成所有 {config.epochs} 个 epoch 的训练")
    return history


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            if images.nelement() == 0 or labels.nelement() == 0:
                continue

            images, labels = images.to(device), labels.to(device)

            # 不计算梯度的前向传播
            with autocast(enabled=config.use_amp):
                outputs = model(images)

                # 添加输出稳定性保护
                outputs = outputs.clamp(min=-20.0, max=20.0)

                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = 100. * correct / total if total > 0 else 0.0

    return val_loss, val_acc, all_preds, all_labels
