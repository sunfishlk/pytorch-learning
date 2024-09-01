import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的线性模型
model = nn.Linear(1, 1)

# 定义损失函数
criterion = nn.MSELoss()

# 使用不同的学习率，学习率越高收敛越快，但是可能导致模型在最优解附近震荡，甚至无法收敛。学习率越低，收敛越慢，更稳定，训练时间长，但是可能会导致模型陷入局部最优解，无法找到全局最优解。
# learning_rates = [0.01]
# learning_rates = [0.05]
# learning_rates = [0.1]
learning_rates = [0.5]


for lr in learning_rates:
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(100):
        inputs = torch.randn(10, 1)
        targets = 2 * inputs + 3

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Learning rate: {lr}, Epoch: {epoch}, Loss: {loss.item()}')