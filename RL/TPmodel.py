import torch
import torch.nn as nn

class GafRegressionCNN(nn.Module):
    def __init__(self, input_channels=12, output_dim=6):
        super(GafRegressionCNN, self).__init__()

        # 卷积层：处理 (12, 60, 60)
        self.features = nn.Sequential(
            # 第一层：捕获 HLOC 间的局部空间相关性
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            # 第二层：下采样
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 60x60 -> 30x30

            # 第三层：深层特征提取
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 30x30 -> 15x15
        )

        # 全局平均池化：增强平移不变性，减少参数
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 回归头：预测 6 个极值
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)  # 输出 6 维线性回归值
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x