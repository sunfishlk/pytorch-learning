import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

# torch
# torchvision
# matplotlib.pyplot

# Hyper Parameters
EPOCH = 1  # 训练轮次
BATCH_SIZE = 64  # 批处理大小
TIME_STEP = 28  # 时间步长
INPUT_SIZE = 28  # 输入尺寸
LR = 0.01  # 学习率

# 训练集
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor()
)

# 测试集
test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=torchvision.transforms.ToTensor()
)
# 取测试集前2000个的测试样本
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255
test_y = test_data.targets.numpy()[:2000]

# DataLoader 数据加载器 在DataLoader中定义训练数据，批处理大小，是否打乱
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

#定义RNN
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(  # 输入28 输出64 使用1层LSTM 因为单纯的RNN效果不好
            input_size=INPUT_SIZE,
            hidden_size=64,   # 28->64
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)  # 64 -> 10 10种分类结果

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        # 取 RNN 输出的最后一个时间步的输出
        out = self.out(r_out[:, -1, :])
        return out

# 实例化    
rnn = RNN()
print(rnn)

# 损失函数和优化器
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

# 训练
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        out = rnn(batch_x)
        loss = loss_func(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # 计算测试集准确率
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# 测试
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

# 画图
plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_data.data[i].numpy(), cmap='gray')
    plt.title('%i' % test_y[i])
    plt.xticks(())
    plt.yticks(())
plt.show()

# RNN处理图像相关的还是有些吃力，图像相关应该还是CNN更合适，RNN适合序列化数据