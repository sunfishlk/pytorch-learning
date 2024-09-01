import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 快速搭建神经网络
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),  # 输入层
    torch.nn.ReLU(),  # 激活函数
    torch.nn.Linear(10, 1)  # 输出层
)

optimizer = torch.optim.SGD(net.parameters(), lr=0.25)
loss_func = torch.nn.MSELoss()

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

plt.ion()
plt.show()

# 训练100次
for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()