import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config import config


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc):
    os.makedirs(config.model_dir, exist_ok=True)
    path = os.path.join(config.model_dir, "best_model.pth")

    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
        "config": vars(config)
    }, path)

    print(f"✅ Model saved at {path}")


def plot_confusion_matrix(labels, preds, title='Confusion Matrix', figsize=(15, 13)):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=figsize)
    hm = sns.heatmap(cm, annot=False, fmt="d", cmap='Blues',
                     cbar=False, robust=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # 保存混淆矩阵
    output_path = os.path.join(config.results_dir, f'{title.replace(" ", "_")}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved to {output_path}")
    return cm


def save_training_report(true_labels, pred_labels, acc, classes=None):
    """保存分类报告和混淆矩阵"""
    os.makedirs(config.results_dir, exist_ok=True)

    # 生成分类报告
    report = classification_report(true_labels, pred_labels, output_dict=True)

    # 保存报告为JSON
    report_path = os.path.join(config.results_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    # 保存为CSV
    class_report = classification_report(true_labels, pred_labels, output_dict=False)
    with open(os.path.join(config.results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Validation Accuracy: {acc:.2f}%\n")
        f.write(class_report)

    print(f"✅ Classification report saved to {report_path}")
    return report


def load_best_model(model, device='cuda'):
    """加载训练好的最佳模型"""
    model_path = os.path.join(config.model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ 模型已加载 | Epoch: {checkpoint['epoch']} | Best Acc: {checkpoint['best_acc']:.2f}%")
    return model
