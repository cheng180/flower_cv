import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetFlower(nn.Module):
    """使用预训练的EfficientNet模型"""

    def __init__(self, num_classes=100):
        super().__init__()
        # 使用预训练的EfficientNet-b3
        self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

        # 冻结前80%的层
        freeze_percentage = 0.8
        total_layers = len(list(self.base_model.children()))
        freeze_index = int(total_layers * freeze_percentage)

        ct = 0
        for child in self.base_model.children():
            if ct < freeze_index:
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1

        # 替换分类头
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1024),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


class FlowerCNN(nn.Module):
    """改进的CNN模型，添加注意力机制"""

    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            self.se_block(32, 16),  # SE注意力模块

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            self.se_block(64, 16),  # SE注意力模块

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            self.se_block(128, 32)  # SE注意力模块
        )

        # 自适应池化
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # 注意力权重
        self.attention_weights = []

    @staticmethod
    def se_block(in_channels, reduction=16):
        """Squeeze-and-Excitation块"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 特征提取
        for layer in self.features:
            if isinstance(layer, nn.Sequential) and len(layer) == 5:  # SE块
                se = layer[0:-1]  # 提取SE模块
                x = se(x) * x  # 应用注意力
            else:
                x = layer(x)

        x = self.pool(x)
        return self.classifier(x)


# 选择要使用的模型
def create_model(model_name, num_classes=100):
    if model_name == 'efficientnet':
        print("Using EfficientNet model")
        return EfficientNetFlower(num_classes)
    else:
        print("Using enhanced CNN model with SE blocks")
        return FlowerCNN(num_classes)
