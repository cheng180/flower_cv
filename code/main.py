import torch
from data_preparation import get_data_loaders
from model import create_model
from train import train_model
from config import config


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 启用cudnn自动调优
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"🆚 CUDA backend: enabled benchmark mode")

    print(f"🖥️  Using device: {device} (CUDA: {torch.cuda.device_count()})")
    if device.type == 'cuda':
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    return device


def main():
    device = setup_device()
    train_loader, val_loader, classes = get_data_loaders()
    config.classes = classes  # 保存类别信息

    print(f"💡 数据集统计: {len(train_loader.dataset)} 训练, {len(val_loader.dataset)} 验证")

    # 创建模型
    model = create_model('efficientnet', num_classes=config.num_classes).to(device)

    # 打印模型概况
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🤖 模型参数: {n_params / 1e6:.2f}M trainable parameters")

    # 开始训练
    train_model(model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
