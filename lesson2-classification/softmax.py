import torch
import torch.nn.functional as F

# 示例输入张量
input_tensor = torch.tensor([1.0, 2.0, 3.0])
print(input_tensor)

# 计算 softmax，F.softmax()返回的是一个张量，其元素值是输入张量的指数值除以所有元素的指数值之和，即归一化后的概率值
softmax_output = F.softmax(input_tensor, dim=0)

print(softmax_output)

pred = torch.max(softmax_output, dim=0)[1]
print(pred)