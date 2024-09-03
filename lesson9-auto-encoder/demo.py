import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

# 超参数
EPOCH = 10
BATCH_SIZE = 64
LR = 0.01

N_TEST_IMG = 5  # 可视化的图片数量

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

# 可视化图片用
test_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor) / 255.

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
            nn.Sigmoid()  # Sigmoid()函数输出在0和1之间
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 实例化
autoencoder = AutoEncoder()

# 优化器和损失
loss_func = torch.nn.MSELoss()  # 均方差损失
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

# 开启动态模式
plt.ion()  # 打开交互模式

# 预先创建一个窗口和子图
fig, axes = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.subplots_adjust(hspace=0.3)

# 训练
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):  # batch_x是data数据集， batch_y是targets标签目标集
        # 展平
        batch_x = batch_x.view(-1, 28*28)
        # 喂数据 算误差
        encoded, decoded = autoencoder(batch_x)
        loss = loss_func(decoded, batch_x)  # 无监督学习，不需要targets标签集，用输入的源数据集和喂给网络后的结果比较算loss
        # 三件套
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train_loss: %.4f' % loss.data.numpy())

            # 可视化测试图片和重建图片
            _, decoded_data = autoencoder(test_data)
            for i in range(N_TEST_IMG):
                axes[0][i].clear()
                axes[0][i].imshow(test_data.data.numpy()[i].reshape(28, 28), cmap='gray')
                axes[0][i].set_xticks(())
                axes[0][i].set_yticks(())
                
                axes[1][i].clear()
                axes[1][i].imshow(decoded_data.data.numpy()[i].reshape(28, 28), cmap='gray')
                axes[1][i].set_xticks(())
                axes[1][i].set_yticks(())

            plt.draw()
            plt.pause(0.1)

plt.ioff()  # 关闭交互模式
plt.show()  # 确保最后一幅图显示
