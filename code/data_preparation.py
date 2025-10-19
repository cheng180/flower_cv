import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 允许加载部分损坏图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 预训练模型的均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FlowerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_train=False):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = os.path.join(self.root_dir, row["filename"])
        label = int(row["label"])

        try:
            image = Image.open(img_name).convert("RGB")
            image = np.array(image)  # 转换为NumPy数组
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            # 对于无效图片，返回一个占位符图片和忽略标签
            placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
            return placeholder, -1

        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image)
            return augmented['image'], label

        return image, label


def extract_data():
    """自动解压 train.zip（仅首次执行）"""
    if not os.path.exists(config.image_dir):
        print(f"[INFO] Extracting {config.raw_data} ...")
        with zipfile.ZipFile(config.raw_data, "r") as zip_ref:
            zip_ref.extractall(config.data_dir)
        print("[INFO] Extraction done.")


def get_transforms():
    """数据增强和预处理管道"""
    # 训练增强
    train_transform = A.Compose([
        A.Resize(512, 512),  # 更大的输入尺寸
        A.RandomResizedCrop(480, 480, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CoarseDropout(max_holes=4, max_height=40, max_width=40, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    # 验证/测试增强
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.CenterCrop(480, 480),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    return train_transform, val_transform


def get_data_loaders():
    """构建训练与验证 DataLoader"""
    extract_data()

    # 创建数据变换
    train_transform, val_transform = get_transforms()

    df = pd.read_csv(config.csv_file)
    print(f"[INFO] Loaded {len(df)} samples from CSV.")

    # 确保标签列存在且为整数
    if "label" not in df.columns:
        df["label"] = df["category_id"].astype("category").cat.codes

    # 检查文件是否存在
    df["exists"] = df["filename"].apply(lambda x: os.path.isfile(os.path.join(config.image_dir, x)))
    missing_count = (~df["exists"]).sum()
    if missing_count > 0:
        print(f"[WARN] {missing_count} files listed in CSV not found on disk. They will be skipped.")
    df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)

    # 类别统计
    classes = sorted(df["label"].unique().tolist())
    print(f"[INFO] Detected {len(classes)} unique classes.")
    print(f"[INFO] Label mapping example: {list(enumerate(classes))[:5]}")

    # 划分训练 / 验证集
    train_df, val_df = train_test_split(
        df, test_size=config.val_split, stratify=df["label"], random_state=42
    )
    print(f"[INFO] Train size: {len(train_df)}, Val size: {len(val_df)}")

    # 保存划分结果
    train_csv = os.path.join(config.data_dir, "train_split.csv")
    val_csv = os.path.join(config.data_dir, "val_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # 构建 Dataset
    train_dataset = FlowerDataset(train_csv, config.image_dir, train_transform, is_train=True)
    val_dataset = FlowerDataset(val_csv, config.image_dir, val_transform)

    print(f"[INFO] Train dataset: {len(train_dataset)} samples")
    print(f"[INFO] Validation dataset: {len(val_dataset)} samples")

    # 自定义 collate_fn 跳过无效样本
    def safe_collate(batch):
        batch = [b for b in batch if b[1] != -1]
        if len(batch) == 0:
            return torch.empty(0), torch.empty(0)
        return torch.utils.data.dataloader.default_collate(batch)

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True, collate_fn=safe_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True, collate_fn=safe_collate
    )

    return train_loader, val_loader, classes
