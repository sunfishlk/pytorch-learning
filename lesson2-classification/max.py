import torch
import torch.nn.functional as F

# 示例输入张量
out = torch.tensor([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]])

# 计算 softmax，返回概率分布张量
softmax_output = F.softmax(out, dim=0)
print("Softmax Output:\n", softmax_output)

# test = torch.tensor([[0.0024, 0.0023, 0.0025],
#         [0.0472, 0.0474, 0.0473],
#         [0.9502, 0.9504, 0.9503]])

# 获取每一行的最大概率对应的类别索引
prediction = torch.max(softmax_output, dim=1)[1]
print("Prediction:\n", prediction)

# 第0维是列，第1维是行