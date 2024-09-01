import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
x,y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# print(x.size())
# print(torch.rand(x.size()))

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(1,10,1)
print(net)

# 损失函数和优化器
optimizer = torch.optim.SGD(net.parameters(),lr=0.2)
loss_func = torch.nn.MSELoss()

plt.ion()   

# 训练
for i in range(100):
    # 喂数据给网络得到预测输出
    prediction = net(x)
    # 计算误差
    loss = loss_func(prediction,y)  # nn.input, target
    # 清空梯度
    optimizer.zero_grad()
    # 误差反向传递
    loss.backward()
    # 优化器步进
    optimizer.step()

    # 可视化训练过程，每训练5次绘制1次图
    if i % 5==0:
        # 清除当前图像
        plt.cla()
        # 绘制散点图，x和y是张量，需要转换为numpy数组
        plt.scatter(x.data.numpy(),y.data.numpy())
        # 绘制预测曲线，红色实现，线宽5
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        # 显示误差值，文本位置0.5 0，设置字体大小20，颜色红色
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        # 暂停0.1秒
        plt.pause(0.1)

plt.ioff()
plt.show()