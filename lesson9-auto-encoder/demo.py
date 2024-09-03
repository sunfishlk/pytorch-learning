#  自编码器 先压缩再解压MNIST数据集

import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

# 超参数
EPOCH = 10
BATCH_SIZE = 64
LR = 0.02

# 训练集
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor()
)

train_loader = Data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=BATCH_SIZE
)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 编码层
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),  # 28x28 -> 128  压缩
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),  # 压缩为3个特征
        )
        # 解码层
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # Sigmod()函数输出为0和1
        )