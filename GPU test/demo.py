import torch

# 检查是否有可用的GPU
gpu_available = torch.cuda.is_available()

print(f"是否有可用的GPU: {gpu_available}")
