import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data 造数据集
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # LongTensor = 64-bit integer

# plt.scatter(x[:, 0].data.numpy(), x[:, 1].data.numpy(), c=y.data.numpy(), cmap='viridis')
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 定义分类网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
    
net = Net(n_feature=2, n_hidden=10, n_output=2)  # 有几个类别就有几个输出
print(net)

# 定义损失函数和优化器
# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

# 可视化
plt.ion()   # 启用交互模式，interactive mode on，实时更新图像
plt.show()  # 交互模式下show不会阻塞，只是刷新窗口，会继续执行后面的代码，非交互模式下show会阻塞，直到关闭图像

# 训练
for i in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if i % 5 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out, dim=0), dim=1)[1]
        # print('softmax:\n', F.softmax(out, dim=0))
        # print('max\n', torch.max(F.softmax(out, dim=0), dim=1))
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, linewidths=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 关闭交互模式
plt.show()  # 阻塞模式下show会阻塞，直到关闭图像，程序结束