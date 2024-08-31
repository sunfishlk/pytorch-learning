import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./number-recognize/data/mnist', train=True,  transform=transform)  # vscode默认的根路径是根目录
test_dataset = datasets.MNIST(root='./number-recognize/data/mnist', train=False,  transform=transform)  # vscode默认的根路径是根目录
# train_dataset = datasets.MNIST(root='./data/mnist', train=True,  transform=transform)  # pycharm默认根路径是当前项目路径
# test_dataset = datasets.MNIST(root='./data/mnist', train=False,  transform=transform)  # pycharm默认根路径是当前项目路径
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.data[i], cmap='gray', interpolation='none')  # data是图像
    plt.title("Labels: {}".format(train_dataset.targets[i]))  # targets是标签
    plt.xticks([])
    plt.yticks([])
plt.show()
