import torch

# 我们已经定义好了模型，对 subject 和 object 进行预测，
# 接下来就是跟目标值进行对比，计算损失了。同时，目标值存在两个不均衡问题，在计算损失时，还需要设置权重参数，对损失值进行适当调节。
true = torch.tensor([[1, 0, 0, 0],[0, 1, 0, 0]])
print(torch.where(true > 0, 1, 0.3)) # 目标序列中，0 的数量远多于 1 的数量，所以对 0 的误差适当降权，争取能匹配出更多实体。
exit()

batch = torch.tensor([
    [[1, 1, 1, 1], [2, 2, 2, 2]],
    [[3, 3, 3, 3], [4, 4, 4, 4]],
    [[5, 5, 5, 5], [6, 6, 6, 6]],
])  # shape:[3, 2, 4]

idx = torch.tensor([[1, 0], [0, 1], [1, 0]])  # shape:[3, 2]
idx = idx.unsqueeze(1)  # shape:[3,1,2]
batch_n = torch.matmul(idx, batch)

print(batch_n.shape) # shape:[3, 1, 4]
print(batch_n + batch) # shape:[3, 2, 4]