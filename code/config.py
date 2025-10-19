import os


class Config:
    # 路径配置
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data", "processed")
    raw_data = os.path.join(project_root, "data", "raw", "train.zip")

    # 数据文件
    csv_file = os.path.join(data_dir, "train_labels.csv")
    image_dir = os.path.join(data_dir, "train")

    # 模型与结果路径
    model_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")

    # 训练参数
    num_classes = 100
    batch_size = 16  # 增加batch size以利用GPU
    num_workers = 8
    lr = 1e-4
    epochs = 50 # 增加训练轮次
    val_split = 0.1

    # 新增的高级训练参数
    weight_decay = 1e-4  # 权重衰减系数
    use_amp = True  # 启用混合精度训练
    label_smoothing = 0.1  # 标签平滑参数
    clip_grad = 5.0  # 梯度裁剪阈值

    # 学习率调度策略
    lr_schedule = 'cosine'  # 'cosine'或'step'
    lr_step_size = 10  # 阶梯下降步长
    lr_gamma = 0.8  # 学习率衰减系数
    warmup_epochs = 3  # 学习率预热轮数
    min_lr = 1e-6  # 最小学习率

    # 混合矩阵相关参数
    plot_confusion_matrix = True
    confusion_matrix_size = (20, 20)


config = Config()
