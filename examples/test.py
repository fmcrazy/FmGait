import torch

# 假设这是两个形状为 (8107,) 的一维张量
tensor1 = torch.randn(8107)
tensor2 = torch.randn(8107)

# 使用 torch.stack() 沿着新的维度进行堆叠（拼接）
stacked_tensor = torch.stack((tensor1, tensor2), dim=1)

# 打印拼接后的张量形状
print(stacked_tensor.shape)