import torch
# import torchvision
import numpy as np
import matplotlib.pyplot as plt

# RNN回归分析 正弦函数 预测余弦 sin->cos

LR = 0.02

# numpy生成数据集
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)  # sin(x) x范围[0,2*pi] 32位浮点
x_np = np.sin(steps)
y_np = np.cos(steps)

# plt.plot(steps, y_np, 'r-', label='target cos(x)')
# plt.plot(steps, x_np, 'b-', label='input sin(x)')
# plt.legend(loc='best')  # 加图例，显示线更清楚
# plt.show()

# 定义网络 RNN 一层rnn 一层out
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=1,  #回归，输入1个曲线，如果是分类，输入是图像就是28
            hidden_size=32,  # 隐藏层32个曲线
            num_layers=1,  # 定义RNN层数，这里为1
            batch_first=True  # batch_first 字段
        )
        self.out = torch.nn.Linear(32, 1)  #线性输出 32->1
    
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
    
        # r_out, h_state = self.rnn(x, h_state)
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs.view(-1, 32, TIME_STEP), h_state
    
rnn =  RNN()
print(rnn)

# 损失函数和优化器
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion()

# 训练
for step in range(100):
    start, end = step*np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, 10, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 画图
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw();
    plt.pause(0.05)

plt.ioff()
plt.show()