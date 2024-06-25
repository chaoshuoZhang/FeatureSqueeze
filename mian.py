import torch

V = torch.ones((128, 3, 32, 32))
importance = torch.ones((128, 3, 32, 32))

# 求和所有通道，获得单个重要性分数 per pixel per image
importance = torch.sum(importance, dim=1, keepdim=True)  # 形状变为 (128, 1, 32, 32)

# 找到每个图像中最不重要的像素的索引
flatten_importance = importance.view(importance.size(0), -1)  # 改变形状为 (128, 1024)
min_indices = torch.argmin(flatten_importance, dim=1)  # 每个图像最小重要性的索引 (128,)

# 更新 V，对每个图像的找到的索引位置的所有通道都置为0
batch_indices = torch.arange(V.size(0)).to(V.device)  # 批次索引 (128,)
channel_indices = torch.tensor([0, 1, 2]).to(V.device)  # 通道索引
height_indices = min_indices // 32  # 行索引
width_indices = min_indices % 32  # 列索引

for i in range(3):  # 遍历每个通道
    V[batch_indices, i, height_indices, width_indices] = 0
print()